@NonCPS
def findDirs(text) {
    def patterns = ['/yocto-builds?[^\\s\'"]*', '/poky[^\\s\'"]*', 
                    'BUILDDIR[\\s=]+[\'"]?([^\\s\'"]+)', 'BUILD_DIR[\\s=]+[\'"]?([^\\s\'"]+)']
    patterns.collectMany { p -> (text =~ p).collect { it instanceof String ? it : it[1] } }
            .unique().findAll { it?.size() > 3 }
}

@NonCPS
def buildJsonArray(list) {
    if (!list || list.size() == 0) return '[]'
    def items = []
    for (int i = 0; i < list.size(); i++) {
        items.add("\"${list[i].replaceAll('"', '\\\\"')}\"")
    }
    return "[${items.join(',')}]"
}

def getPipelineJobs() {
    Jenkins.instance.getAllItems(org.jenkinsci.plugins.workflow.job.WorkflowJob.class).collect { it.fullName }
}

def getPipelineScript(jobName) {
    def job = Jenkins.instance.getItemByFullName(jobName, org.jenkinsci.plugins.workflow.job.WorkflowJob.class)
    if (!job) error("Job not found")
    def definition = job.getDefinition()
    if (definition instanceof org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition) return definition.getScript()
    if (definition instanceof org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition) return "// SCM: ${definition.getScriptPath()}"
    return "// Unable to extract"
}

pipeline {
    agent any
    parameters {
        choice(name: 'RAG_MODEL', choices: ['codellama:7b'])
        string(name: 'RAG_API_URL', defaultValue: 'http://localhost:8000')
    }
    environment {
        RAG_API_URL = "${params.RAG_API_URL}"
    }
    options { skipDefaultCheckout(); timeout(time: 2, unit: 'HOURS') }
    
    stages {
        stage('Select Pipeline') {
            steps {
                script {
                    def pipelines = getPipelineJobs()
                    if (!pipelines) error("No pipelines found")
                    env.TARGET_PIPELINE = input(message: 'Select pipeline', ok: 'Analyze',
                        parameters: [choice(name: 'Pipelines', choices: pipelines.join('\n'))])
                }
            }
        }
        
        stage('Extract & Analyze') {
            steps {
                script {
                    env.PIPELINE_SCRIPT = getPipelineScript(env.TARGET_PIPELINE)
                    env.BUILD_DIRECTORY = findDirs(env.PIPELINE_SCRIPT)?.getAt(0) ?: '/tmp'
                    env.DISK_SPACE_GB = sh(script: "df -h ${env.BUILD_DIRECTORY} 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//' || echo '50'",
                        returnStdout: true).trim()
                }
            }
        }
        
        stage('RAG Prediction') {
            steps {
                script {
                    env.PREDICTION_ID = sh(script: 'cat /proc/sys/kernel/random/uuid', returnStdout: true).trim()
                    
                    def query = """Analyze pipeline. Predict SUCCESS/FAIL/HIGH-RISK with confidence %.
Disk: ${env.DISK_SPACE_GB}GB
Pipeline: ${env.TARGET_PIPELINE}
${env.PIPELINE_SCRIPT}"""
                    
                    writeFile(file: 'req.json', text: """{"model":"${params.RAG_MODEL}","stream":false,"prediction_id":"${env.PREDICTION_ID}","messages":[{"role":"user","content":"${query.replaceAll('"', '\\\\"').replaceAll('\n', '\\\\n')}"}]}""")
                    
                    echo "Calling RAG API with Prediction ID: ${env.PREDICTION_ID}"
                    
                    def resp = sh(script: '''curl -sX POST -H "Content-Type: application/json" -H "X-User-ID: jenkins" \
                        --data @req.json -w "\\nHTTP:%{http_code}" --max-time 1200 "$RAG_API_URL/v1/chat/completions"''', 
                        returnStdout: true).trim()
                    
                    def parts = resp.split('HTTP:')
                    if (parts.length < 2 || parts[1].trim() != '200') error("RAG API failed")
                    
                    def json = readJSON(text: parts[0])
                    
                    // ALWAYS use backend's prediction_id
                    if (json.prediction_id) {
                        echo "Backend stored prediction_id: ${json.prediction_id}"
                        env.PREDICTION_ID = json.prediction_id
                        echo "Using backend prediction_id for feedback: ${env.PREDICTION_ID}"
                    } else {
                        echo "WARNING: Backend didn't return prediction_id, using: ${env.PREDICTION_ID}"
                    }
                    
                    def analysis = json.choices[0].message.content
                    echo "=" * 60 + "\nRAG ANALYSIS\n" + "=" * 60 + "\n${analysis}\n" + "=" * 60
                    writeFile(file: 'analysis.txt', text: analysis)
                    archiveArtifacts artifacts: 'analysis.txt'
                    
                    def predMatch = (analysis =~ /PREDICTION:\s*(SUCCESS|FAIL|HIGH-RISK|PASS)/)
                    env.PREDICTION_RESULT = predMatch ? predMatch[0][1] : 
                        (analysis.toUpperCase().contains('HIGH-RISK') || analysis.toUpperCase().contains('HIGH_RISK') ? 'HIGH-RISK' : 
                         analysis.toUpperCase().contains('FAIL') ? 'FAIL' : 'PASS')
                    
                    env.CONFIDENCE_SCORE = ((analysis =~ /CONFIDENCE:\s*(\d+)%/) ? (analysis =~ /CONFIDENCE:\s*(\d+)%/)[0][1] : '50')
                    echo "Result: ${env.PREDICTION_RESULT} @ ${env.CONFIDENCE_SCORE}%"
                }
            }
        }
        
        stage('Approve Execution') {
            steps {
                script {
                    if (env.PREDICTION_RESULT == 'PASS' && env.CONFIDENCE_SCORE.toInteger() >= 80) {
                        env.EXECUTION_DECISION = 'AUTO'
                        env.APPROVER = 'System'
                        echo "Auto-approved (${env.CONFIDENCE_SCORE}% >= 80%)"
                    } else {
                        echo "Manual approval required: ${env.PREDICTION_RESULT} @ ${env.CONFIDENCE_SCORE}%"
                        timeout(time: 15, unit: 'MINUTES') {
                            env.EXECUTION_DECISION = input(message: "Proceed with ${env.PREDICTION_RESULT}?", 
                                parameters: [choice(name: 'Do want to deploy?', choices: ['Yes', 'No'])])
                            env.APPROVER = 'Manual'
                        }
                        if (env.EXECUTION_DECISION == 'No') currentBuild.result = 'ABORTED'
                    }
                }
            }
        }
        
        stage('Execute Pipeline') {
            when { expression { env.EXECUTION_DECISION != 'No' } }
            steps { script { build job: env.TARGET_PIPELINE, wait: false, propagate: false } }
        }
        
        stage('Collect Feedback') {
            steps {
                script {
                    catchError(buildResult: currentBuild.result ?: 'SUCCESS', stageResult: 'SUCCESS') {
                        echo "=" * 80
                        echo "HELP IMPROVE AI PREDICTIONS"
                        echo "=" * 80
                        echo "Prediction: ${env.PREDICTION_RESULT} @ ${env.CONFIDENCE_SCORE}%"
                        echo "Your feedback helps the AI learn!"
                        echo "=" * 80
                        
                        try {
                            timeout(time: 5, unit: 'MINUTES') {
                                def wantFeedback = input(
                                    message: 'Provide feedback?',
                                    ok: 'Yes',
                                    parameters: [choice(name: 'Choice', choices: ['Yes', 'Skip'])]
                                )
                                
                                if (wantFeedback == 'Yes') {
                                    timeout(time: 10, unit: 'MINUTES') {
                                        def fb = input(
                                            message: 'Feedback Details',
                                            ok: 'Submit',
                                            parameters: [
                                                choice(name: 'ACTUAL', choices: ['SUCCESS', 'FAILURE']),
                                                string(name: 'CONF', defaultValue: ''),
                                                text(name: 'MISSED', defaultValue: ''),
                                                text(name: 'FALSE', defaultValue: ''),
                                                text(name: 'COMMENTS', defaultValue: '')
                                            ]
                                        )
                                        
                                        env.ACTUAL_RESULT = fb.ACTUAL
                                        if (fb.CONF?.trim()) env.CORRECTED_CONFIDENCE = fb.CONF.trim()
                                        if (fb.MISSED?.trim()) env.MISSED_ISSUES = fb.MISSED
                                        if (fb.FALSE?.trim()) env.FALSE_ALARMS = fb.FALSE
                                        if (fb.COMMENTS?.trim()) env.USER_COMMENTS = fb.COMMENTS
                                        echo "Feedback collected: ${env.ACTUAL_RESULT}"
                                    }
                                } else {
                                    echo "Feedback skipped"
                                }
                            }
                        } catch (e) {
                            echo "Feedback timeout"
                        }
                    }
                }
            }
        }
    }
    
    post {
        always {
            script {
                echo "=" * 80 + "\nSUMMARY\n" + "=" * 80
                echo "Pipeline: ${env.TARGET_PIPELINE}"
                echo "Prediction: ${env.PREDICTION_RESULT} @ ${env.CONFIDENCE_SCORE}%"
                echo "Prediction ID: ${env.PREDICTION_ID}"
                echo "Decision: ${env.EXECUTION_DECISION} by ${env.APPROVER ?: 'System'}"
                echo "Disk: ${env.DISK_SPACE_GB}GB on ${env.BUILD_DIRECTORY}"
                echo "=" * 80
                
                def actual = env.ACTUAL_RESULT ?: (currentBuild.result == 'SUCCESS' ? 'SUCCESS' : 'FAILURE')
                
                def confValue = null
                if (env.CORRECTED_CONFIDENCE && env.CORRECTED_CONFIDENCE.trim()) {
                    try {
                        confValue = env.CORRECTED_CONFIDENCE.trim().toInteger()
                        if (confValue < 0 || confValue > 100) confValue = null
                    } catch (e) { confValue = null }
                }
                
                def missedList = []
                if (env.MISSED_ISSUES) {
                    def lines = env.MISSED_ISSUES.split('\n')
                    for (int i = 0; i < lines.length; i++) {
                        def line = lines[i].trim()
                        if (line && !line.startsWith('-')) missedList.add(line)
                        else if (line.startsWith('- ')) missedList.add(line.substring(2).trim())
                    }
                }
                
                def falseList = []
                if (env.FALSE_ALARMS) {
                    def lines = env.FALSE_ALARMS.split('\n')
                    for (int i = 0; i < lines.length; i++) {
                        def line = lines[i].trim()
                        if (line && !line.startsWith('-')) falseList.add(line)
                        else if (line.startsWith('- ')) falseList.add(line.substring(2).trim())
                    }
                }
                
                def missedJson = buildJsonArray(missedList)
                def falseJson = buildJsonArray(falseList)
                def confJson = confValue != null ? confValue.toString() : 'null'
                def commentsText = (env.USER_COMMENTS ?: 'Automatic feedback').replaceAll('"', '\\\\"').replaceAll('\n', ' ')
                def fbType = env.ACTUAL_RESULT ? 'manual' : 'automatic'
                
                def payload = """{"prediction_id":"${env.PREDICTION_ID}","actual_build_result":"${actual}","corrected_confidence":${confJson},"missed_issues":${missedJson},"false_positives":${falseJson},"user_comments":"${commentsText}","feedback_type":"${fbType}"}"""
                
                writeFile(file: 'fb.json', text: payload)
                
                echo "Sending feedback with prediction_id: ${env.PREDICTION_ID}"
                
                try {
                    def fbResp = sh(script: '''curl -sX POST -H "Content-Type: application/json" -H "X-User-ID: jenkins" \
                        --data @fb.json -w "\\nHTTP:%{http_code}" "$RAG_API_URL/api/feedback/submit"''', returnStdout: true).trim()
                    
                    def fbParts = fbResp.split('HTTP:')
                    if (fbParts.length >= 2) {
                        def httpCode = fbParts[1].trim()
                        if (httpCode == '200' || httpCode == '201') {
                            echo "✓ Feedback sent successfully (HTTP ${httpCode})"
                        } else {
                            echo "Feedback failed (HTTP ${httpCode})"
                        }
                    }
                } catch (e) {
                    echo " Feedback error: ${e.message}"
                }
                
                sh 'rm -f req.json fb.json analysis.txt'
            }
        }
    }
}
