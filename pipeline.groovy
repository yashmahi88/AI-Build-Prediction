// looks for common directory patterns like /yocto-builds, /poky, BUILDDIR, etc.
// returns: The directory path, or /tmp if nothing found
@NonCPS
def findBuildDir(text) {
    ['/yocto-builds?[^\\s\'"]*', '/poky[^\\s\'"]*', 'BUILDDIR[\\s=]+[\'"]?([^\\s\'"]+)', 'BUILD_DIR[\\s=]+[\'"]?([^\\s\'"]+)']
        .collectMany { p -> (text =~ p).collect { it instanceof String ? it : it[1] } }
        .unique().findAll { it?.size() > 3 }?.getAt(0) ?: '/tmp'
}

@NonCPS
def parseList(text) { text ? text.split('\n')*.trim().findAll { it && !it.startsWith('-') } : [] } // takes text with line breaks and converts to array, removes dashes

@NonCPS
def extractPrediction(text) {
    def match = text =~ /PREDICTION:\s*(PASS|FAIL|HIGH-RISK)/
    match ? match[0][1] : (text.toUpperCase() =~ /(HIGH-RISK|HIGH_RISK)/ ? 'HIGH-RISK' : text.toUpperCase().contains('FAIL') ? 'FAIL' : 'PASS')   // Looks for PASS, FAIL, or HIGH-RISK in the AI's text
}

@NonCPS
def extractConfidence(text) { 
    def match = text =~ /CONFIDENCE:\s*(\d+)%/   // Looks for "CONFIDENCE:" pattern in text
    match ? match[0][1] : ''  // return empty strng if not found
}

def getPipelines() { Jenkins.instance.getAllItems(org.jenkinsci.plugins.workflow.job.WorkflowJob.class)*.fullName } // get list of all available pipelines on the node

def getPipelineScript(name) {     // to get script of the pipline
    def job = Jenkins.instance.getItemByFullName(name, org.jenkinsci.plugins.workflow.job.WorkflowJob.class)  
    if (!job) error("Pipeline not found: ${name}")
    def definition = job.getDefinition()
    definition instanceof org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition ? definition.getScript() : "// SCM: ${definition?.getScriptPath()}"
}

def callAPI(endpoint, payload) {   // to call the AI API for prediction or feedback
    writeFile file: 'api.json', text: payload
    def resp = sh(script: "curl -sX POST '${env.RAG_API_URL}${endpoint}' -H 'Content-Type: application/json' -H 'X-User-ID: jenkins' --data @api.json -w '\\nHTTP:%{http_code}' --max-time 1200", returnStdout: true).trim()
    def parts = resp.split('HTTP:')
    [ok: parts.size() >= 2 && parts[1].trim() == '200', code: parts.size() >= 2 ? parts[1].trim() : '000', body: parts[0]]
}

pipeline {
    agent any
    parameters {
        // choice(name: 'RAG_MODEL', choices: ['codellama:7b','mistral:7b-instruct-q4_0','qwen2.5:1.5b'])   // which AI model to use for prediction
        string(name: 'RAG_API_URL', defaultValue: 'http://localhost:8000')  // url where AI service is running
        booleanParam(name: 'WAIT_FOR_BUILD', defaultValue: true, description: 'Wait for build completion') // wait for build to finish before asking for feedback
    }
    environment {
        RAG_API_URL = "${params.RAG_API_URL}"   
        AUTO_THRESHOLD = '80'   // threshold to auto approve if confidence is >= to this value
        DISK_CRITICAL = '60'   // fail is disk space is <60 GB as yocto requires more space
        DISK_OPTIMAL = '100'   // optimal disk space value
    }
    options { skipDefaultCheckout(); timeout(time: 8, unit: 'HOURS'); timestamps(); buildDiscarder(logRotator(numToKeepStr: '50')) }  // timeout,add time to console output, keep only last 50 build
    
    stages {
        stage('Select') {
            steps {
                script {
                    def pipes = getPipelines()  // get all pipelines
                    if (!pipes) error("No pipelines found")
                    env.TARGET = input(message: 'Select pipeline', ok: 'Analyze', parameters: [choice(name: 'Pipeline', choices: pipes.join('\n'))])  //user selects pipeline
                    echo "→ ${env.TARGET}"
                }
            }
        }
        
        stage('Extract') {
            steps {
                script {
                    env.SCRIPT = getPipelineScript(env.TARGET)   //get script
                    env.BUILD_DIR = findBuildDir(env.SCRIPT)   // get build directory
                    env.DISK_GB = sh(script: "df -h '${env.BUILD_DIR}' 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//' || echo 50", returnStdout: true).trim()   
                     // get disk space in GB
                    def disk = env.DISK_GB.toInteger()
                    if (disk < env.DISK_CRITICAL.toInteger()) error(" CRITICAL: ${disk}GB < ${env.DISK_CRITICAL}GB minimum required") //Stop immediately if disk space too low
                    env.DISK_OK = disk >= env.DISK_OPTIMAL.toInteger() ? 'true' : 'false'     //check is it optimal or caution               
                    echo "→ ${env.BUILD_DIR} (${disk}GB - ${env.DISK_OK == 'true' ? 'OPTIMAL' : 'CAUTION'})"
                }
            }
        }
        
        stage('Analyze') {
            steps {
                script {
                    env.PRED_ID = sh(script: 'uuidgen || cat /proc/sys/kernel/random/uuid', returnStdout: true).trim()  //generate unique id to store prediction
                    def query = "Analyze pipeline. Predict PASS/FAIL/HIGH-RISK with confidence %.\nDisk: ${env.DISK_GB}GB\nPipeline: ${env.TARGET}\n${env.SCRIPT}"  // que for AI 
                    def resp = callAPI('/v1/chat/completions', groovy.json.JsonOutput.toJson([stream: false, prediction_id: env.PRED_ID, messages: [[role: 'user', content: query]]])) //cal AI API
                    if (!resp.ok) error("API failed: HTTP ${resp.code}")  //check if AI failed or succeeded
                    
                    def json = readJSON(text: resp.body)  //parse AI response
                    if (json.prediction_id) env.PRED_ID = json.prediction_id
                    
                    def analysis = json.choices[0].message.content   // get AI analysis text and uplaod as artifact
                    def banner = '=' * 80
                    echo "\n${banner}\n${analysis}\n${banner}"
                    writeFile file: 'analysis.txt', text: analysis
                    archiveArtifacts artifacts: 'analysis.txt', allowEmptyArchive: true
                    
                    env.PRED = extractPrediction(analysis)  // extract prediction from AI response
                    env.CONF = extractConfidence(analysis) ?: '0'  // extract confidence from AI response
                    echo "→ ${env.PRED} @ ${env.CONF}%"
                }
            }
        }
        
        stage('Approve') {
            steps {
                script {
                    def conf = env.CONF.toInteger()
                    def autoApprove = env.PRED == 'PASS' && conf >= env.AUTO_THRESHOLD.toInteger() && env.DISK_OK == 'true'  //auto apprve only if conditions are met
                    
                    if (autoApprove) {
                        env.DECISION = 'AUTO'; env.APPROVER = 'System'
                        echo " Auto-approved: PASS @ ${conf}% + ${env.DISK_GB}GB"
                    } else {        //manual approval required if conditions are not met 
                        def reasons = []
                        if (env.PRED != 'PASS') reasons << "Prediction=${env.PRED}"
                        if (conf < env.AUTO_THRESHOLD.toInteger()) reasons << "Confidence=${conf}%<${env.AUTO_THRESHOLD}%"
                        if (env.DISK_OK == 'false') reasons << "Disk=${env.DISK_GB}GB<${env.DISK_OPTIMAL}GB"
                        echo " Manual approval: ${reasons.join(', ')}"
                        
                        try {
                            timeout(time: 15, unit: 'MINUTES') {    //wait 15 min for user decision
                                env.DECISION = input(message: "Proceed? ${env.PRED}@${conf}% (${env.DISK_GB}GB)", ok: 'Yes', parameters: [choice(name: 'Decision', choices: ['Yes', 'No'])])  //as the user for decison
                                env.APPROVER = 'Manual'
                            }
                            if (env.DECISION == 'No') { echo " Rejected"; currentBuild.result = 'ABORTED' }   //aborted if user selects NO
                        } catch (e) {
                            echo " Timeout/Abort"
                            env.DECISION = 'ABORTED'; env.APPROVER = 'Timeout'; currentBuild.result = 'ABORTED'
                        }
                    }
                }
            }
        }
        
        stage('Execute') {
            when { expression { env.DECISION in ['AUTO', 'Yes'] } }   //only run this if decision was AUTO OR Yes
            steps {
                script {
                    try {
                        def b = build(job: env.TARGET, wait: params.WAIT_FOR_BUILD, propagate: false)  //trigger the seelcted pipeline
                        if (params.WAIT_FOR_BUILD) {
                            env.RESULT = b.result; env.NUM = b.number.toString() //save the actual result
                            echo " Build #${env.NUM}: ${env.RESULT}"
                        } else { env.RESULT = 'TRIGGERED' }
                    } catch (Exception e) { echo " ${e.message}"; env.RESULT = 'ERROR' }   //build failed to start
                }
            }
        }
        
        stage('Feedback') {  //ask for feedback
            when { expression { (env.DECISION in ['AUTO', 'Yes'] && params.WAIT_FOR_BUILD && env.RESULT != 'ERROR') || env.DECISION == 'No' } }  // build ran and completed or, user rejected the build
            steps {
                script {
                    catchError(buildResult: currentBuild.result ?: 'SUCCESS', stageResult: 'SUCCESS') {  
                        def banner = '=' * 80
                        echo "\n${banner}\nFEEDBACK\n${banner}\n${env.PRED}@${env.CONF}% | ${env.RESULT ?: 'Not executed'}\n${banner}"
                        try {
                            timeout(time: 30, unit: 'MINUTES') {  // wait 30 mins for user to provide feedback 
                                if (input(message: 'Provide feedback?', ok: 'Yes', parameters: [choice(name: 'Provide Feedback', choices: ['Yes', 'Skip'])]) == 'Yes') {
                                    def params = [string(name: 'CONF', defaultValue: ''), text(name: 'MISSED', defaultValue: ''), text(name: 'FALSE', defaultValue: ''), text(name: 'COMMENTS', defaultValue: '')]  //build feedback form
                                    if (env.RESULT && env.RESULT != 'TRIGGERED') params.add(0, choice(name: 'ACTUAL', choices: ['SUCCESS', 'FAILURE']))
                                    
                                    //save feedback to env vars
                                    def fb = input(message: 'Feedback', ok: 'Submit', parameters: params)
                                    if (fb.ACTUAL) env.ACTUAL = fb.ACTUAL
                                    if (fb.CONF?.trim()) env.CONF_CORR = fb.CONF.trim()
                                    if (fb.MISSED?.trim()) env.MISSED = fb.MISSED
                                    if (fb.FALSE?.trim()) env.FALSE = fb.FALSE
                                    if (fb.COMMENTS?.trim()) env.COMMENTS = fb.COMMENTS
                                    echo "✓ Collected"
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
            script {    //print sumamry
                def banner = '=' * 80
                echo "\n${banner}\nSUMMARY\n${banner}\n${env.TARGET}\n${env.PRED}@${env.CONF}% | ${env.DECISION ?: 'N/A'} by ${env.APPROVER ?: 'System'}\n${env.RESULT ? "Build: ${env.RESULT}" : ''} | Disk: ${env.DISK_GB}GB\nID: ${env.PRED_ID}\n${banner}"
                 //send feedback to AI system
                if (env.DECISION in ['AUTO', 'Yes', 'No'] && env.DECISION != 'ABORTED') {
                    def confVal = null
                    if (env.CONF_CORR?.trim()) {  //get the confidence value if user provided
                        try { confVal = env.CONF_CORR.trim().toInteger(); confVal = (confVal >= 0 && confVal <= 100) ? confVal : null } catch (e) {}
                    }
                    
                    def resp = callAPI('/api/feedback/submit', groovy.json.JsonOutput.toJson([
                        prediction_id: env.PRED_ID,   //get prediction id
                        actual_build_result: env.ACTUAL ?: (env.RESULT ?: 'NOT_EXECUTED'),  //what was actual result
                        corrected_confidence: confVal,  //user's corrected confidence
                        missed_issues: parseList(env.MISSED ?: ''),   //issues that AI missed
                        false_positives: parseList(env.FALSE ?: ''),  //false warnings if any give by AI
                        user_comments: env.COMMENTS ?: (env.DECISION == 'No' ? 'User rejected' : 'Auto'),  //comments by user
                        feedback_type: (env.ACTUAL || env.COMMENTS || env.CONF_CORR) ? 'manual' : 'automatic'
                    ]))
                    echo resp.ok ? "✓ Feedback sent" : " Failed (HTTP ${resp.code})"
                }
                sh 'rm -f api.json analysis.txt 2>/dev/null || true'
            }
        }
    }
}