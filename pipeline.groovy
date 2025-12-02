@NonCPS
def findBuildDir(text) { // To fetch the build directory from pipeline script content (fallback to /tmp if not found)
    ['/yocto-builds?[^\\s\'"]*', '/poky[^\\s\'"]*', 'BUILDDIR[\\s=]+[\'"]?([^\\s\'"]+)', 'BUILD_DIR[\\s=]+[\'"]?([^\\s\'"]+)']
        .collectMany { p -> (text =~ p).collect { it instanceof String ? it : it[1] } } // Gather regex matches for dir
        .unique().findAll { it?.size() > 3 }?.getAt(0) ?: '/tmp' // Remove duplicates, filter out short/empty, pick first or default
}

@NonCPS
def parseList(text) { text ? text.split('\n')*.trim().findAll { it && !it.startsWith('-') } : [] } // Split multiline text to trimmed list, removing empties/dashes

@NonCPS
def extractPrediction(text) { // Pull prediction result from text, try to infer if missing
    def match = text =~ /PREDICTION:\s*(PASS|FAIL|HIGH-RISK)/
    match ? match[0][1] : (text.toUpperCase() =~ /(HIGH-RISK|HIGH_RISK)/ ? 'HIGH-RISK' : text.toUpperCase().contains('FAIL') ? 'FAIL' : 'PASS')
}

@NonCPS
def extractConfidence(text) { // Pull confidence percentage from text, if available
    def match = text =~ /CONFIDENCE:\s*(\d+)%/
    match ? match[0][1] : ''
}

def getPipelines() { Jenkins.instance.getAllItems(org.jenkinsci.plugins.workflow.job.WorkflowJob.class)*.fullName } // Get all workflow pipeline job names

def getPipelineScript(name) {  // Get pipeline Groovy script or SCM path for a given job
    def job = Jenkins.instance.getItemByFullName(name, org.jenkinsci.plugins.workflow.job.WorkflowJob.class)
    if (!job) error("Pipeline not found: ${name}")
    def definition = job.getDefinition()
    definition instanceof org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition ? definition.getScript() : "// SCM: ${definition?.getScriptPath()}"
}

def callAPI(endpoint, payload) { // Post JSON to remote RAG API, get response and status
    writeFile file: 'api.json', text: payload
    def resp = sh(script: "curl -sX POST '${env.RAG_API_URL}${endpoint}' -H 'Content-Type: application/json' -H 'X-User-ID: jenkins' --data @api.json -w '\\nHTTP:%{http_code}' --max-time 1200", returnStdout: true).trim()
    def parts = resp.split('HTTP:')
    [ok: parts.size() >= 2 && parts[1].trim() == '200', code: parts.size() >= 2 ? parts[1].trim() : '000', body: parts[0]]
}

pipeline {
    agent any // Run controller stages on any available Jenkins worker node
    parameters {
        string(name: 'RAG_API_URL', defaultValue: 'http://localhost:8000') // Parameter for base RAG API URL
        booleanParam(name: 'WAIT_FOR_BUILD', defaultValue: true) // Parameter for waiting for the build to finish
    }
    environment {
        RAG_API_URL = "${params.RAG_API_URL}" // Use parameter value in env
        AUTO_THRESHOLD = '80' // Confidence threshold for auto-approve
        DISK_CRITICAL = '60' // Minimum disk space (GB) to not error
        DISK_OPTIMAL = '100' // Minimum disk for auto-approve (GB)
    }
    options {
        skipDefaultCheckout(); // Don't fetch SCM for this controller job
        timeout(time: 8, unit: 'HOURS'); // Pipeline maximum time
        timestamps(); // Print time with all log lines
        buildDiscarder(logRotator(numToKeepStr: '50')) // Only keep 50 builds
    }

    stages {
        stage('Select') {
            steps {
                script {
                    def pipes = getPipelines() // List all jobs
                    if (!pipes) error("No pipelines found")
                    env.TARGET = input(message: 'Select pipeline', ok: 'Analyze', parameters: [choice(name: 'Pipeline', choices: pipes.join('\n'))]) // Ask user to pick a pipeline
                    echo "→ ${env.TARGET}" // Log picked pipeline name
                }
            }
        }

        stage('Extract') {
            steps {
                script {
                    env.SCRIPT = getPipelineScript(env.TARGET) // Get script content for chosen pipeline
                    env.BUILD_DIR = findBuildDir(env.SCRIPT) // Try to find build dir from script
                    env.DISK_GB = sh(script: "df -h '${env.BUILD_DIR}' 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//' || echo 50", returnStdout: true).trim() // Get free GB for that directory (default 50 if fails)
                    def disk = env.DISK_GB.toInteger() // Parse as integer
                    if (disk < env.DISK_CRITICAL.toInteger()) error("CRITICAL: ${disk}GB < ${env.DISK_CRITICAL}GB minimum required") // Hard fail below critical
                    env.DISK_OK = disk >= env.DISK_OPTIMAL.toInteger() ? 'true' : 'false' // Is optimal?
                    echo "→ ${env.BUILD_DIR} (${disk}GB - ${env.DISK_OK == 'true' ? 'OPTIMAL' : 'CAUTION'})" // Print result
                }
            }
        }

        stage('Analyze') {
            steps {
                script {
                    env.PRED_ID = sh(script: 'uuidgen || cat /proc/sys/kernel/random/uuid', returnStdout: true).trim() // Random prediction ID
                    def query = """Analyze this Yocto/Jenkins pipeline for build success prediction:
Disk: ${env.DISK_GB}GB available
Target: ${env.TARGET}
Build dir: ${env.BUILD_DIR}


SCRIPT TO ANALYZE:
${env.SCRIPT}


Use ALL relevant documentation, patterns, and best practices from your knowledge base.""" // Natural-language context for LLM
                    def resp = callAPI('/v1/chat/completions', groovy.json.JsonOutput.toJson([stream: false, prediction_id: env.PRED_ID, messages: [[role: 'user', content: query]]])) // Call chat API
                    if (!resp.ok) error("API failed: HTTP ${resp.code}") // Fail if not 200

                    def json = readJSON(text: resp.body) // Parse JSON reply
                    if (json.prediction_id) env.PRED_ID = json.prediction_id // Use server's ID if given

                    def analysis = json.choices[0].message.content // Get only string result
                    def banner = '=' * 80 // Big separator
                    echo "\n${banner}\n${analysis}\n${banner}" // Print result for human review
                    writeFile file: 'analysis.txt', text: analysis // Save for download
                    archiveArtifacts artifacts: 'analysis.txt', allowEmptyArchive: true // Archive file for Jenkins browser

                    env.PRED = extractPrediction(analysis) // Pull pass/fail/high-risk
                    env.CONF = extractConfidence(analysis) ?: '0' // Pull confidence number
                    echo "→ ${env.PRED} @ ${env.CONF}%" // Log both for user
                }
            }
        }

        stage('Approve') {
            steps {
                script {
                    def conf = env.CONF.toInteger() // Parse confidence as integer
                    def autoApprove = env.PRED == 'PASS' && conf >= env.AUTO_THRESHOLD.toInteger() && env.DISK_OK == 'true' // Decide auto-approve
                    if (autoApprove) {
                        env.DECISION = 'AUTO'; env.APPROVER = 'System' // Mark as automatically approved
                        echo "Auto-approved: PASS @ ${conf}% + ${env.DISK_GB}GB"
                    } else {
                        def reasons = []
                        if (env.PRED != 'PASS') reasons << "Prediction=${env.PRED}"
                        if (conf < env.AUTO_THRESHOLD.toInteger()) reasons << "Confidence=${conf}%<${env.AUTO_THRESHOLD}%"
                        if (env.DISK_OK == 'false') reasons << "Disk=${env.DISK_GB}GB<${env.DISK_OPTIMAL}GB"
                        echo "Manual approval: ${reasons.join(', ')}" // Log why not auto
                        try {
                            timeout(time: 15, unit: 'MINUTES') {
                                env.DECISION = input(message: "Proceed? ${env.PRED}@${conf}% (${env.DISK_GB}GB)", ok: 'Yes', parameters: [choice(name: 'Decision', choices: ['Yes', 'No'])])
                                env.APPROVER = 'Manual'
                            }
                            if (env.DECISION == 'No') { echo "Rejected"; currentBuild.result = 'ABORTED' }
                        } catch (e) {
                            echo "Timeout/Abort"
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
                        def b = build(job: env.TARGET, wait: params.WAIT_FOR_BUILD, propagate: false) // Call selected job
                        if (params.WAIT_FOR_BUILD) {
                            env.RESULT = b.result; env.NUM = b.number.toString()
                            echo "Build #${env.NUM}: ${env.RESULT}"
                        } else { env.RESULT = 'TRIGGERED' }
                    } catch (Exception e) { echo "${e.message}"; env.RESULT = 'ERROR' }
                }
            }
        }

        stage('Feedback') {
            when { expression { (env.DECISION in ['AUTO', 'Yes'] && params.WAIT_FOR_BUILD && env.RESULT != 'ERROR') || env.DECISION == 'No' } }
            steps {
                script {
                    catchError(buildResult: currentBuild.result ?: 'SUCCESS', stageResult: 'SUCCESS') {
                        def banner = '=' * 80
                        echo "\n${banner}\nFEEDBACK\n${banner}\n${env.PRED}@${env.CONF}% | ${env.RESULT ?: 'Not executed'}\n${banner}"
                        try {
                            timeout(time: 30, unit: 'MINUTES') {
                                if (input(message: 'Provide feedback?', ok: 'Yes', parameters: [choice(name: 'Provide Feedback', choices: ['Yes', 'Skip'])]) == 'Yes') {
                                    def params = [string(name: 'CONF', defaultValue: ''), text(name: 'MISSED', defaultValue: ''), text(name: 'FALSE', defaultValue: ''), text(name: 'COMMENTS', defaultValue: '')]
                                    if (env.RESULT && env.RESULT != 'TRIGGERED') params.add(0, choice(name: 'ACTUAL', choices: ['SUCCESS', 'FAILURE']))
                                    def fb = input(message: 'Feedback', ok: 'Submit', parameters: params)
                                    if (fb.ACTUAL) env.ACTUAL = fb.ACTUAL
                                    if (fb.CONF?.trim()) env.CONF_CORR = fb.CONF.trim()
                                    if (fb.MISSED?.trim()) env.MISSED = fb.MISSED
                                    if (fb.FALSE?.trim()) env.FALSE = fb.FALSE
                                    if (fb.COMMENTS?.trim()) env.COMMENTS = fb.COMMENTS
                                    echo "✓ Collected"
                                }
                            }
                        } catch (e) { echo "Skipped" }
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                def banner = '=' * 80
                echo "\n${banner}\nSUMMARY\n${banner}\n${env.TARGET}\n${env.PRED}@${env.CONF}% | ${env.DECISION ?: 'N/A'} by ${env.APPROVER ?: 'System'}\n${env.RESULT ? "Build: ${env.RESULT}" : ''} | Disk: ${env.DISK_GB}GB\nID: ${env.PRED_ID}\n${banner}"
                if (env.DECISION in ['AUTO', 'Yes', 'No'] && env.DECISION != 'ABORTED') {
                    def confVal = null
                    if (env.CONF_CORR?.trim()) {
                        try {
                            confVal = env.CONF_CORR.trim().toInteger();
                            confVal = (confVal >= 0 && confVal <= 100) ? confVal : null
                        } catch (e) {}
                    }
                    def resp = callAPI('/api/feedback/submit', groovy.json.JsonOutput.toJson([
                        prediction_id: env.PRED_ID,
                        actual_build_result: env.ACTUAL ?: (env.RESULT ?: 'NOT_EXECUTED'),
                        corrected_confidence: confVal,
                        missed_issues: parseList(env.MISSED ?: ''),
                        false_positives: parseList(env.FALSE ?: ''),
                        user_comments: env.COMMENTS ?: (env.DECISION == 'No' ? 'User rejected' : 'Auto'),
                        feedback_type: (env.ACTUAL || env.COMMENTS || env.CONF_CORR) ? 'manual' : 'automatic'
                    ]))
                    echo resp.ok ? "✓ Feedback sent" : "Failed (HTTP ${resp.code})"
                }
                sh 'rm -f api.json analysis.txt 2>/dev/null || true'
            }
        }
    }
}
