@NonCPS
def findBuildDir(text) {
    ['/yocto-builds?[^\\s\'"]*', '/poky[^\\s\'"]*', 'BUILDDIR[\\s=]+[\'"]?([^\\s\'"]+)', 'BUILD_DIR[\\s=]+[\'"]?([^\\s\'"]+)']
        .collectMany { p -> (text =~ p).collect { it instanceof String ? it : it[1] } }
        .unique().findAll { it?.size() > 3 }?.getAt(0) ?: '/tmp'
}

@NonCPS
def parseList(text) { text ? text.split('\n')*.trim().findAll { it && !it.startsWith('-') } : [] }

@NonCPS
def extractPrediction(text) {
    def match = text =~ /PREDICTION:\s*(PASS|FAIL|HIGH-RISK)/
    match ? match[0][1] : (text.toUpperCase() =~ /(HIGH-RISK|HIGH_RISK)/ ? 'HIGH-RISK' : text.toUpperCase().contains('FAIL') ? 'FAIL' : 'PASS')
}

@NonCPS
def extractConfidence(text) { ((text =~ /CONFIDENCE:\s*(\d+)%/) ? (text =~ /CONFIDENCE:\s*(\d+)%/)[0][1] : '50') }

def getPipelines() { Jenkins.instance.getAllItems(org.jenkinsci.plugins.workflow.job.WorkflowJob.class)*.fullName }

def getPipelineScript(name) {
    def job = Jenkins.instance.getItemByFullName(name, org.jenkinsci.plugins.workflow.job.WorkflowJob.class)
    if (!job) error("Pipeline not found: ${name}")
    def definition = job.getDefinition()
    definition instanceof org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition ? definition.getScript() : "// SCM: ${definition?.getScriptPath()}"
}

def callAPI(endpoint, payload) {
    writeFile file: 'api.json', text: payload
    def resp = sh(script: "curl -sX POST '${env.RAG_API_URL}${endpoint}' -H 'Content-Type: application/json' -H 'X-User-ID: jenkins' --data @api.json -w '\\nHTTP:%{http_code}' --max-time 1200", returnStdout: true).trim()
    def parts = resp.split('HTTP:')
    [ok: parts.size() >= 2 && parts[1].trim() == '200', code: parts.size() >= 2 ? parts[1].trim() : '000', body: parts[0]]
}

pipeline {
    agent any
    parameters {
        choice(name: 'RAG_MODEL', choices: ['codellama:7b'])
        string(name: 'RAG_API_URL', defaultValue: 'http://localhost:8000')
        booleanParam(name: 'WAIT_FOR_BUILD', defaultValue: true, description: 'Wait for build completion')
    }
    environment {
        RAG_API_URL = "${params.RAG_API_URL}"
        AUTO_THRESHOLD = '80'
    }
    options { skipDefaultCheckout(); timeout(time: 8, unit: 'HOURS'); timestamps(); buildDiscarder(logRotator(numToKeepStr: '50')) }
    
    stages {
        stage('Select') {
            steps {
                script {
                    def pipes = getPipelines()
                    if (!pipes) error("No pipelines found")
                    env.TARGET = input(message: 'Select pipeline', ok: 'Analyze', parameters: [choice(name: 'Pipeline', choices: pipes.join('\n'))])
                    echo "→ ${env.TARGET}"
                }
            }
        }
        
        stage('Extract') {
            steps {
                script {
                    env.SCRIPT = getPipelineScript(env.TARGET)
                    env.BUILD_DIR = findBuildDir(env.SCRIPT)
                    env.DISK_GB = sh(script: "df -h '${env.BUILD_DIR}' 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//' || echo 50", returnStdout: true).trim()
                    echo "→ ${env.BUILD_DIR} (${env.DISK_GB}GB)"
                }
            }
        }
        
        stage('Analyze') {
            steps {
                script {
                    env.PRED_ID = sh(script: 'uuidgen || cat /proc/sys/kernel/random/uuid', returnStdout: true).trim()
                    
                    def query = "Analyze pipeline. Predict PASS/FAIL/HIGH-RISK with confidence %.\nDisk: ${env.DISK_GB}GB\nPipeline: ${env.TARGET}\n${env.SCRIPT}"
                    def payload = groovy.json.JsonOutput.toJson([model: params.RAG_MODEL, stream: false, prediction_id: env.PRED_ID, messages: [[role: 'user', content: query]]])
                    
                    def resp = callAPI('/v1/chat/completions', payload)
                    if (!resp.ok) error("API failed: HTTP ${resp.code}")
                    
                    def json = readJSON(text: resp.body)
                    if (json.prediction_id) env.PRED_ID = json.prediction_id
                    
                    def analysis = json.choices[0].message.content
                    def banner = '=' * 80
                    echo "\n${banner}\n${analysis}\n${banner}"
                    writeFile file: 'analysis.txt', text: analysis
                    archiveArtifacts artifacts: 'analysis.txt', allowEmptyArchive: true
                    
                    env.PRED = extractPrediction(analysis)
                    env.CONF = extractConfidence(analysis)
                    echo " ${env.PRED} @ ${env.CONF}%"
                }
            }
        }
        
        stage('Approve') {
            steps {
                script {
                    def conf = env.CONF.toInteger()
                    if (env.PRED == 'PASS' && conf >= env.AUTO_THRESHOLD.toInteger()) {
                        env.DECISION = 'AUTO'; env.APPROVER = 'System'
                        echo " Auto-approved (${conf}% ≥ ${env.AUTO_THRESHOLD}%)"
                    } else {
                        echo " ${env.PRED} @ ${conf}%"
                        try {
                            timeout(time: 15, unit: 'MINUTES') {
                                env.DECISION = input(message: "Proceed with ${env.PRED}?", ok: 'Yes', parameters: [choice(name: 'Trigger Analyzed Pipeline?', choices: ['Yes', 'No'])])
                                env.APPROVER = 'Manual'
                            }
                            if (env.DECISION == 'No') { echo " Rejected"; currentBuild.result = 'ABORTED' }
                        } catch (e) {
                            echo " Aborted/Timeout"
                            env.DECISION = 'ABORTED'; env.APPROVER = 'Timeout'; currentBuild.result = 'ABORTED'
                        }
                    }
                }
            }
        }
        
        stage('Execute') {
            when { expression { env.DECISION in ['AUTO', 'Yes'] } }
            steps {
                script {
                    try {
                        def b = build(job: env.TARGET, wait: params.WAIT_FOR_BUILD, propagate: false)
                        if (params.WAIT_FOR_BUILD) {
                            env.RESULT = b.result; env.NUM = b.number.toString()
                            echo " Build #${env.NUM}: ${env.RESULT}"
                        } else {
                            env.RESULT = 'TRIGGERED'
                        }
                    } catch (Exception e) {
                        echo " ${e.message}"; env.RESULT = 'ERROR'
                    }
                }
            }
        }
        
        stage('Feedback') {
            when { expression { (env.DECISION in ['AUTO', 'Yes'] && params.WAIT_FOR_BUILD && env.RESULT != 'ERROR') || env.DECISION == 'No' } }
            steps {
                script {
                    catchError(buildResult: currentBuild.result ?: 'SUCCESS', stageResult: 'SUCCESS') {
                        def banner = '=' * 80
                        echo "\n${banner}\nFEEDBACK\n${banner}\n${env.PRED} @ ${env.CONF}%\n${env.RESULT ?: 'Not executed'}\n${banner}"
                        try {
                            timeout(time: 30, unit: 'MINUTES') {
                                if (input(message: 'Provide feedback?', ok: 'Yes', parameters: [choice(name: 'Provide Feedback?', choices: ['Yes', 'Skip'])]) == 'Yes') {
                                    def params = [
                                        string(name: 'CONF', defaultValue: ''), 
                                        text(name: 'MISSED', defaultValue: ''),
                                        text(name: 'FALSE', defaultValue: ''), 
                                        text(name: 'COMMENTS', defaultValue: '')
                                    ]
                                    if (env.RESULT && env.RESULT != 'TRIGGERED') {
                                        params.add(0, choice(name: 'ACTUAL', choices: ['SUCCESS', 'FAILURE']))
                                    }
                                    def fb = input(message: 'Feedback', ok: 'Submit', parameters: params)
                                    if (fb.ACTUAL) env.ACTUAL = fb.ACTUAL
                                    if (fb.CONF?.trim()) env.CONF_CORR = fb.CONF.trim()
                                    if (fb.MISSED?.trim()) env.MISSED = fb.MISSED
                                    if (fb.FALSE?.trim()) env.FALSE = fb.FALSE
                                    if (fb.COMMENTS?.trim()) env.COMMENTS = fb.COMMENTS
                                    echo " Feeback Collected"
                                }
                            }
                        } catch (e) { echo " Skipped" }
                    }
                }
            }
        }
    }
    
    post {
        always {
            script {
                def banner = '=' * 80
                echo "\n${banner}\nSUMMARY\n${banner}\n${env.TARGET}\n${env.PRED} @ ${env.CONF}%\n${env.DECISION ?: 'N/A'} by ${env.APPROVER ?: 'System'}\n${env.RESULT ? "Result: ${env.RESULT}" : ''}\nID: ${env.PRED_ID}\n${banner}"
                
                if (env.DECISION in ['AUTO', 'Yes', 'No'] && env.DECISION != 'ABORTED') {
                    def confVal = null
                    if (env.CONF_CORR?.trim()) {
                        try { def c = env.CONF_CORR.trim().toInteger(); confVal = (c >= 0 && c <= 100) ? c : null } catch (e) {}
                    }
                    
                    def payload = groovy.json.JsonOutput.toJson([
                        prediction_id: env.PRED_ID,
                        actual_build_result: env.ACTUAL ?: (env.RESULT ?: 'NOT_EXECUTED'),
                        corrected_confidence: confVal,
                        missed_issues: parseList(env.MISSED ?: ''),
                        false_positives: parseList(env.FALSE ?: ''),
                        user_comments: env.COMMENTS ?: (env.DECISION == 'No' ? 'User rejected' : 'Auto'),
                        feedback_type: (env.ACTUAL || env.COMMENTS || env.CONF_CORR) ? 'manual' : 'automatic'
                    ])
                    
                    def resp = callAPI('/api/feedback/submit', payload)
                    echo resp.ok ? "✓ Feedback sent" : " Failed (HTTP ${resp.code})"
                }
                sh 'rm -f api.json analysis.txt 2>/dev/null || true'
            }
        }
    }
}
