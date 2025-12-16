import jenkins.model.Jenkins

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

def controllerLabel = Jenkins.get().getComputer("")?.node?.labelString ?: "built-in" // Get controller node label or default to built-in
def getPipelines() { Jenkins.instance.getAllItems(org.jenkinsci.plugins.workflow.job.WorkflowJob.class)*.fullName } // Get all workflow pipeline job names

def getPipelineScript(name) { // Get pipeline Groovy script or SCM path for a given job
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
        string(name: 'RAG_API_URL', defaultValue: 'http://localhost:8000', description: 'RAG API URL (optional)') // Parameter for base RAG API URL
        choice(name: 'SELECTED_NODE', choices: ['any', controllerLabel] + Jenkins.instance.nodes.collect { it.nodeName }.findAll { it }, description: 'Node to run disk check and build') // Parameter for node selection
        string(name: 'BUILD_DIR', defaultValue: '', description: 'Build directory path (must be accessible on the selected node)') // Parameter for build directory path
        booleanParam(name: 'WAIT_FOR_BUILD', defaultValue: true, description: 'Wait for build to finish') // Parameter for waiting for the build to finish
    }
    environment {
        RAG_API_URL = "${params.RAG_API_URL}" // Use parameter value in env
        AUTO_THRESHOLD = '80' // Confidence threshold for auto-approve
        DISK_CRITICAL = '60' // Minimum disk space (GB) to not error
        DISK_OPTIMAL = '100' // Minimum disk for auto-approve (GB)
    }
    options {
        skipDefaultCheckout() // Don't fetch SCM for this controller job
        timeout(time: 8, unit: 'HOURS') // Pipeline maximum time
        timestamps() // Print time with all log lines
        buildDiscarder(logRotator(numToKeepStr: '50')) // Only keep 50 builds
    }

    stages {
        stage('Select Pipeline') { // First stage: choose which pipeline to analyze
            steps {
                script {
                    def pipes = getPipelines() // List all jobs
                    if (!pipes) error("No pipelines found")
                    env.TARGET = input(message: 'Select pipeline', ok: 'Analyze', parameters: [choice(name: 'Pipeline', choices: pipes.join('\n'))]) // Ask user to pick a pipeline
                    echo "→ Selected pipeline: ${env.TARGET}" // Log picked pipeline name
                }
            }
        }

        stage('Validate Build Directory') { // Second stage: check disk space on selected node
            agent { label params.SELECTED_NODE == 'any' ? '' : params.SELECTED_NODE } // Run on selected node
            steps {
                script {
                    if (!params.BUILD_DIR?.trim()) error("Build directory parameter 'BUILD_DIR' must not be empty") // Fail if empty
                    env.BUILD_DIR = params.BUILD_DIR.trim() // Store trimmed value
                    
                    // Capture the actual node name where this stage runs
                    env.ACTUAL_NODE = env.NODE_NAME ?: 'master' // Store actual node name for consistency
                    echo "→ Running on node: ${env.ACTUAL_NODE}" // Log actual node
                    echo "→ Using build directory: ${env.BUILD_DIR}" // Log directory

                    env.DISK_GB = sh( // Run df to get available disk space
                        script: "df -BG '${env.BUILD_DIR}' 2>/dev/null | awk 'NR==2 {print \$4}' | sed 's/G//'",
                        returnStdout: true
                    ).trim()
                    def disk = env.DISK_GB.isInteger() ? env.DISK_GB.toInteger() : 0 // Parse as integer
                    if (disk == 0) error("Could not determine disk space available for build directory '${env.BUILD_DIR}'") // Fail if unable to check
                    if (disk < env.DISK_CRITICAL.toInteger()) error("CRITICAL: ${disk}GB < ${env.DISK_CRITICAL}GB minimum required") // Hard fail below critical
                    env.DISK_OK = disk >= env.DISK_OPTIMAL.toInteger() ? 'true' : 'false' // Is optimal?

                    echo "→ Disk space available: ${disk}GB - ${env.DISK_OK == 'true' ? 'OPTIMAL' : 'CAUTION'}" // Print result
                    env.DISK_GB = disk.toString() // Store as string
                }
            }
        }

        stage('Analyze') { // Third stage: call RAG API to analyze pipeline and predict result
            agent any // Run on any node
            steps {
                script {
                    env.SCRIPT = getPipelineScript(env.TARGET) // Get script content for chosen pipeline
                    env.PRED_ID = sh(script: 'uuidgen || cat /proc/sys/kernel/random/uuid', returnStdout: true).trim() // Random prediction ID
                    def query = """Analyze this Yocto/Jenkins pipeline for build success prediction:
Disk: ${env.DISK_GB}GB available
Target: ${env.TARGET}
Build dir: ${env.BUILD_DIR}
Node: ${env.ACTUAL_NODE}

SCRIPT TO ANALYZE:
${env.SCRIPT}

Use ALL relevant documentation, patterns, and best practices from your knowledge base.""" // Natural-language context for LLM
                    def resp = callAPI('/v1/chat/completions', groovy.json.JsonOutput.toJson([ // Call chat API
                        stream: false,
                        prediction_id: env.PRED_ID,
                        messages: [[role: 'user', content: query]]
                    ]))
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
                    echo "→ Prediction: ${env.PRED} @ ${env.CONF}%" // Log both for user
                }
            }
        }

        stage('Approve') { // Fourth stage: decide whether to proceed with build
            agent any // Run on any node
            steps {
                script {
                    def conf = env.CONF.toInteger() // Parse confidence as integer
                    def autoApprove = env.PRED == 'PASS' && conf >= env.AUTO_THRESHOLD.toInteger() && env.DISK_OK == 'true' // Decide auto-approve

                    if (autoApprove) { // If all conditions met
                        env.DECISION = 'AUTO' // Mark as automatically approved
                        env.APPROVER = 'System' // Approver is system
                        echo "Auto-approved: PASS @ ${conf}% + ${env.DISK_GB}GB disk space"
                    } else { // Otherwise require manual approval
                        def reasons = [] // Collect reasons for manual decision
                        if (env.PRED != 'PASS') reasons << "Prediction=${env.PRED}"
                        if (conf < env.AUTO_THRESHOLD.toInteger()) reasons << "Confidence=${conf}% < ${env.AUTO_THRESHOLD}%"
                        if (env.DISK_OK == 'false') reasons << "Disk=${env.DISK_GB}GB < ${env.DISK_OPTIMAL}GB"
                        echo "Manual approval required: ${reasons.join(', ')}" // Log why not auto

                        try { // Wrap manual input in try to handle timeout/abort
                            timeout(time: 15, unit: 'MINUTES') { // Limit approval wait time
                                env.DECISION = input( // Prompt user whether to proceed
                                    message: "Proceed? Prediction: ${env.PRED} @ ${conf}% confidence, Disk: ${env.DISK_GB}GB",
                                    ok: 'Yes',
                                    parameters: [choice(name: 'Decision', choices: ['Yes', 'No'])]
                                )
                                env.APPROVER = 'Manual' // Mark approver as manual user
                            }
                            if (env.DECISION == 'No') { // If user chooses No
                                echo "User rejected build." // Log rejection
                                currentBuild.result = 'ABORTED' // Mark build as aborted
                            }
                        } catch (e) { // Handle timeout or user abort
                            echo "Timeout or abort during approval." // Log that approval was not completed
                            env.DECISION = 'ABORTED' // Set decision to aborted
                            env.APPROVER = 'Timeout' // Mark approver as timeout
                            currentBuild.result = 'ABORTED' // Abort the pipeline
                        }
                    }
                }
            }
        }

        stage('Execute') { // Fifth stage: actually trigger the selected pipeline build
            when { expression { env.DECISION in ['AUTO', 'Yes'] } } // Only run if decision is AUTO or Yes
            agent { label env.ACTUAL_NODE } // Run on the SAME node as disk check stage
            steps {
                script {
                    echo "→ Executing on node: ${env.NODE_NAME} (same as validation)" // Confirm same node
                    try { // Wrap downstream build in try to capture errors
                        def b = build(job: env.TARGET, wait: params.WAIT_FOR_BUILD, propagate: false) // Call selected job
                        if (params.WAIT_FOR_BUILD) { // If waiting for build completion
                            env.RESULT = b.result // Store downstream build result
                            env.NUM = b.number.toString() // Store downstream build number as string
                            echo "Build #${env.NUM} completed with status: ${env.RESULT}" // Log downstream build result
                        } else {
                            env.RESULT = 'TRIGGERED' // If not waiting, mark as triggered only
                        }
                    } catch (Exception e) { // Handle any failure to trigger or run downstream build
                        echo "Build trigger failed: ${e.message}" // Log error message
                        env.RESULT = 'ERROR' // Mark result as error
                    }
                }
            }
        }

        stage('Feedback') { // Sixth stage: optionally collect feedback from user about prediction quality
            when { expression { (env.DECISION in ['AUTO', 'Yes'] && params.WAIT_FOR_BUILD && env.RESULT != 'ERROR') || env.DECISION == 'No' } } // Only run when either build ran or user rejected
            agent any // Run on any node
            steps {
                script {
                    catchError(buildResult: currentBuild.result ?: 'SUCCESS', stageResult: 'SUCCESS') { // Ensure feedback stage failures do not fail whole build
                        def banner = '=' * 80 // Separator line for readability
                        echo "\n${banner}\nFEEDBACK\n${banner}\nPrediction: ${env.PRED} @ ${env.CONF}% | Build Result: ${env.RESULT ?: 'Not executed'}\n${banner}" // Print quick summary for feedback context
                        try { // Wrap interactive feedback in timeout
                            timeout(time: 30, unit: 'MINUTES') { // Give user up to 30 minutes for feedback
                                if (input(message: 'Provide feedback?', ok: 'Yes', parameters: [choice(name: 'Provide Feedback', choices: ['Yes', 'Skip'])]) == 'Yes') { // Ask user whether to provide feedback
                                    def params = [ // Prepare basic feedback fields
                                        string(name: 'CONF', defaultValue: ''),
                                        text(name: 'MISSED', defaultValue: ''),
                                        text(name: 'FALSE', defaultValue: ''),
                                        text(name: 'COMMENTS', defaultValue: '')
                                    ]
                                    if (env.RESULT && env.RESULT != 'TRIGGERED') { // If build actually ran and finished
                                        params.add(0, choice(name: 'ACTUAL', choices: ['SUCCESS', 'FAILURE'])) // Add actual outcome choice in front
                                    }

                                    def fb = input(message: 'Feedback', ok: 'Submit', parameters: params) // Ask user to fill feedback form
                                    if (fb.ACTUAL) env.ACTUAL = fb.ACTUAL // Save actual build result if provided
                                    if (fb.CONF?.trim()) env.CONF_CORR = fb.CONF.trim() // Save corrected confidence if given
                                    if (fb.MISSED?.trim()) env.MISSED = fb.MISSED // Save missed issues text if given
                                    if (fb.FALSE?.trim()) env.FALSE = fb.FALSE // Save false positive text if given
                                    if (fb.COMMENTS?.trim()) env.COMMENTS = fb.COMMENTS // Save user comments if given
                                    echo "✓ Feedback collected" // Log that feedback was collected
                                }
                            }
                        } catch (e) { // Handle skipped, timeout, or cancel of feedback
                            echo "Feedback skipped or timeout" // Log that feedback stage was skipped
                        }
                    }
                }
            }
        }
    }

    post { // Post section runs after all stages
        always { // Always run this block regardless of build result
            script {
                def banner = '=' * 80 // Build separator line
                echo "\n${banner}\nSUMMARY\n${banner}" // Print summary header
                echo "Pipeline: ${env.TARGET}" // Log target pipeline
                echo "Prediction: ${env.PRED} @ ${env.CONF}%" // Log prediction and confidence
                echo "Decision: ${env.DECISION ?: 'N/A'} by ${env.APPROVER ?: 'System'}" // Log decision and approver
                if (env.RESULT) echo "Build Result: ${env.RESULT}" // Log build result if available
                echo "Disk Space: ${env.DISK_GB}GB" // Log disk space
                echo "Node Used: ${env.ACTUAL_NODE}" // Log actual node used
                echo "Prediction ID: ${env.PRED_ID}" // Log prediction ID
                echo "${banner}" // Close summary

                if (env.DECISION in ['AUTO', 'Yes', 'No'] && env.DECISION != 'ABORTED') { // Only send feedback if a meaningful decision happened
                    def confVal = null // Placeholder for numeric corrected confidence
                    if (env.CONF_CORR?.trim()) { // If corrected confidence text exists
                        try {
                            confVal = env.CONF_CORR.trim().toInteger() // Try to parse as integer
                            confVal = (confVal >= 0 && confVal <= 100) ? confVal : null // Accept only 0–100, else discard
                        } catch (e) { /* ignore */ } // Ignore parse errors
                    }

                    def resp = callAPI('/api/feedback/submit', groovy.json.JsonOutput.toJson([ // Send feedback payload back to RAG feedback endpoint
                        prediction_id: env.PRED_ID,
                        actual_build_result: env.ACTUAL ?: (env.RESULT ?: 'NOT_EXECUTED'),
                        corrected_confidence: confVal,
                        missed_issues: parseList(env.MISSED ?: ''),
                        false_positives: parseList(env.FALSE ?: ''),
                        user_comments: env.COMMENTS ?: (env.DECISION == 'No' ? 'User rejected' : 'Auto'),
                        feedback_type: (env.ACTUAL || env.COMMENTS || env.CONF_CORR) ? 'manual' : 'automatic'
                    ]))
                    echo resp.ok ? "✓ Feedback sent" : "Failed to send feedback (HTTP ${resp.code})" // Log feedback submission status
                }

                sh 'rm -f api.json analysis.txt 2>/dev/null || true' // Clean up temporary files used for API and analysis
            }
        }
    }
}