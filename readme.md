# AI-Powered Yocto Build Prediction Pipeline - Documentation

## Table of Contents
1. [Overview](#overview)
2. [What This Pipeline Does](#what-this-pipeline-does)
3. [How It Works](#how-it-works)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Using the Pipeline](#using-the-pipeline)
8. [Understanding the Flow](#understanding-the-flow)
9. [Feedback System](#feedback-system)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

***

## Overview

### What is This Pipeline?

This is an **intelligent Jenkins pipeline** that uses AI to predict whether a Yocto build will succeed or fail **before** running it. It saves time and resources by:

- **Analyzing build scripts** using AI (RAG + LLM)
- **Predicting outcomes** with confidence scores
- **Auto-approving safe builds** based on prediction confidence
- **Learning from feedback** to improve future predictions
- **Checking system resources** (disk space) before execution

### Key Benefits

- ⏱️ **Save Time**: Know build outcome before running (1-2 minutes vs hours)
- 💰 **Save Resources**: Skip builds predicted to fail
- 🎯 **High Accuracy**: AI learns from feedback to improve predictions
- 🤖 **Auto-Approval**: Automatically runs high-confidence PASS predictions
- 📊 **Detailed Analysis**: Get insights on why a build might fail

***

## What This Pipeline Does

### Simple Explanation

Imagine you have a **smart assistant** that:

1. **Reads** your Yocto build pipeline script
2. **Analyzes** it using knowledge from past builds
3. **Predicts** if it will succeed or fail (with confidence %)
4. **Checks** if you have enough disk space
5. **Auto-approves** if confidence is high, or **asks you** to approve
6. **Runs** the actual build if approved
7. **Learns** from your feedback to get smarter

### Real-World Scenario

```
Before (Without AI):
User clicks "Build" → Wait 2 hours → Build fails → Waste of time

After (With AI):
User runs this pipeline → AI predicts FAIL (85% confidence) in 2 minutes
→ User sees why it will fail → Fixes issue → Runs again → SUCCESS
Total time saved: ~2 hours
```

***

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Jenkins AI Prediction Pipeline             │
└─────────────────┬───────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┐
    │             │             │              │
    ▼             ▼             ▼              ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐
│ Select  │ │ Extract │ │ Analyze  │ │  Approve   │
│Pipeline │ │ Script  │ │ with AI  │ │ (Auto/Man) │
└─────────┘ └─────────┘ └──────────┘ └────────────┘
                              │              │
                              ▼              ▼
                      ┌──────────────┐ ┌──────────┐
                      │  RAG API     │ │ Execute  │
                      │ (Port 8000)  │ │  Build   │
                      └──────────────┘ └──────────┘
                              │              │
                              ▼              ▼
                      ┌──────────────┐ ┌──────────┐
                      │ Vectorstore  │ │ Feedback │
                      │  + LLM       │ │ to AI    │
                      └──────────────┘ └──────────┘
```

### Pipeline Stages Explained

#### 1. **Select Stage**
- Lists all available Jenkins pipelines on your system
- User selects which pipeline to analyze
- Uses Jenkins API to fetch pipeline list

#### 2. **Extract Stage**
- Gets the selected pipeline's Groovy script
- Searches for build directory paths (e.g., `/yocto-builds`, `/poky`)
- Checks available disk space
- Validates minimum disk space requirement (60GB)

#### 3. **Analyze Stage**
- Sends pipeline script + context to RAG API
- AI analyzes the script using vectorstore knowledge
- Returns prediction: **PASS**, **FAIL**, or **HIGH-RISK**
- Provides confidence score (0-100%)
- Generates detailed analysis report

#### 4. **Approve Stage**
- **Auto-Approval** if:
  - Prediction = PASS
  - Confidence ≥ 80%
  - Disk space ≥ 100GB
- **Manual Approval** otherwise:
  - Shows prediction and reasons
  - User decides: Yes or No
  - 15-minute timeout for decision

#### 5. **Execute Stage**
- Only runs if approved (auto or manual)
- Triggers the actual Yocto build pipeline
- Waits for completion (if configured)
- Captures actual build result

#### 6. **Feedback Stage**
- Prompts user for feedback after build completes
- Collects:
  - Actual result (if different from prediction)
  - Corrected confidence
  - Missed issues AI didn't catch
  - False positives AI warned about
- Sends feedback to AI system for learning

***

## Prerequisites

### System Requirements

```
Jenkins Server:
- Jenkins 2.300+
- Jenkins Plugins:
  - Pipeline (workflow-aggregator)
  - Pipeline: Groovy
  - Build Timeout
- Java 11+

RAG API Server:
- Python 3.10+
- FastAPI application running on port 8000
- Ollama with models loaded
- PostgreSQL database
```

### Network Requirements

```
Jenkins → RAG API: http://localhost:8000 (or remote IP)
Port 8000 must be accessible from Jenkins server
```

***

## Installation & Setup

### Step 1: Verify RAG API is Running

Before setting up the pipeline, ensure your RAG API is operational:

```bash
# Check if API is accessible
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "vectorstore_loaded": true
}
```

If not running, refer to the main RAG system documentation to start it.

### Step 2: Create Jenkins Pipeline Job

1. **Log into Jenkins** (http://your-jenkins:8080)

2. **Create New Item**:
   - Click "New Item"
   - Enter name: `AI-Yocto-Predictor`
   - Select: "Pipeline"
   - Click "OK"

3. **Configure Pipeline**:
   - Scroll to "Pipeline" section
   - Definition: "Pipeline script"
   - Paste the entire pipeline code

4. **Save** the configuration

### Step 3: Configure Pipeline Parameters

The pipeline has default parameters you can customize:

```groovy
parameters {
    string(name: 'RAG_API_URL', defaultValue: 'http://localhost:8000')
    booleanParam(name: 'WAIT_FOR_BUILD', defaultValue: true)
}
```

**To change defaults:**
- Edit the pipeline script
- Modify `defaultValue` fields
- Save

### Step 4: Set Up Environment Variables

The pipeline uses these environment variables:

```groovy
environment {
    RAG_API_URL = "${params.RAG_API_URL}"
    AUTO_THRESHOLD = '80'        // Auto-approve if confidence >= 80%
    DISK_CRITICAL = '60'         // Fail if disk < 60GB
    DISK_OPTIMAL = '100'         // Optimal disk space
}
```

**Customization:**
- `AUTO_THRESHOLD`: Increase to 90 for stricter auto-approval
- `DISK_CRITICAL`: Increase to 80 if your builds need more space
- `DISK_OPTIMAL`: Set to your typical free space

### Step 5: Test the Setup

Run a test to verify everything works:

```bash
# 1. Open the pipeline in Jenkins
# 2. Click "Build Now"
# 3. Follow the interactive prompts
# 4. Verify AI prediction appears in console output
```

**Expected output:**
```
===============================================================================
Analyze pipeline. Predict PASS/FAIL with confidence.
...
PREDICTION: PASS
CONFIDENCE: 85%
...
===============================================================================
```

***

## Configuration

### Essential Settings

#### 1. RAG API URL

**Default:** `http://localhost:8000`

**Change if:**
- RAG API runs on different server: `http://192.168.1.100:8000`
- Using different port: `http://localhost:9000`

```groovy
string(name: 'RAG_API_URL', defaultValue: 'http://your-server:8000')
```

#### 2. Wait for Build Completion

**Default:** `true` (pipeline waits for build to finish)

**Set to `false` if:**
- Build takes many hours
- You don't need immediate feedback
- Running multiple builds in parallel

```groovy
booleanParam(name: 'WAIT_FOR_BUILD', defaultValue: false)
```

#### 3. Auto-Approval Threshold

**Default:** 80% confidence

**Adjust based on your risk tolerance:**

```groovy
// Conservative (safer, more manual approvals)
AUTO_THRESHOLD = '90'

// Aggressive (faster, more auto-approvals)
AUTO_THRESHOLD = '70'

// Very conservative (almost always manual)
AUTO_THRESHOLD = '95'
```

#### 4. Disk Space Requirements

**Defaults:**
- Critical: 60GB minimum (build fails if less)
- Optimal: 100GB (required for auto-approval)

**For larger Yocto builds:**

```groovy
DISK_CRITICAL = '80'    // Minimum 80GB
DISK_OPTIMAL = '150'    // Prefer 150GB for auto-approval
```

***

## Using the Pipeline

### Basic Workflow

#### Step 1: Start the Pipeline

1. Navigate to the pipeline in Jenkins
2. Click **"Build Now"** or **"Build with Parameters"**
3. Set parameters (or use defaults)
4. Click **"Build"**

#### Step 2: Select Target Pipeline

```
Console Output:
Input requested
Select pipeline
[Choice Parameter] Pipeline
  - Yocto-Build-Pipeline
  - Yocto-Integration-Build
  - Custom-Recipe-Build
```

**Action:** Select the pipeline you want to analyze

#### Step 3: Review AI Analysis

```
===============================================================================
ANALYSIS REPORT

Pipeline: Yocto-Build-Pipeline
Disk Space: 95GB (CAUTION)

PREDICTION: HIGH-RISK
CONFIDENCE: 72%

REASON:
- Missing DEPENDS += "openssl-native" (detected in line 45)
- Recipe version conflict: systemd 250 vs required 249
- Potential bitbake parsing error in custom.bbclass

RECOMMENDATION:
Add missing dependencies and resolve version conflicts before building.
===============================================================================

→ HIGH-RISK @ 72%
```

#### Step 4: Approval Decision

**Auto-Approval (if conditions met):**
```
✓ Auto-approved: PASS @ 85% + 105GB
```

**Manual Approval:**
```
Input requested
Proceed? HIGH-RISK@72% (95GB)
[Choice Parameter] Decision
  - Yes
  - No

Reason for manual approval:
  - Prediction=HIGH-RISK
  - Confidence=72%<80%
  - Disk=95GB<100GB
```

**Action:** 
- Choose **Yes** to proceed with build
- Choose **No** to abort

#### Step 5: Build Execution (if approved)

```
Triggering: Yocto-Build-Pipeline
Waiting for completion...
Build #42: SUCCESS
```

#### Step 6: Provide Feedback

```
===============================================================================
FEEDBACK
===============================================================================
HIGH-RISK@72% | SUCCESS
===============================================================================

Input requested
Provide feedback?
[Choice Parameter] Provide Feedback
  - Yes
  - Skip
```

**If you select "Yes":**

```
Feedback Form:
[Choice] ACTUAL: SUCCESS / FAILURE
[String] CONF: (corrected confidence 0-100)
[Text] MISSED: Issues AI missed (one per line)
[Text] FALSE: False warnings from AI (one per line)
[Text] COMMENTS: Additional notes
```

**Example feedback:**

```
ACTUAL: SUCCESS
CONF: 85
MISSED:
  - Build succeeded despite version warning
  - systemd 250 is actually compatible
FALSE:
  - bitbake parsing error (false alarm)
COMMENTS: AI was too cautious, build worked fine
```

***

## Understanding the Flow

### Decision Tree

```
Start Pipeline
    ↓
Select Target Pipeline
    ↓
Get Script + Check Disk
    ↓
Send to AI for Analysis
    ↓
AI Returns: PASS/FAIL/HIGH-RISK with Confidence
    ↓
Is it PASS + ≥80% + ≥100GB disk?
    ├─ YES → Auto-Approve → Execute Build
    └─ NO  → Manual Approval Required
              ├─ User says YES → Execute Build
              └─ User says NO  → Abort (send feedback)
                   ↓
Build Completes (if executed)
    ↓
Ask for Feedback
    ↓
Send Feedback to AI
    ↓
AI Learns & Improves
```

### Auto-Approval Logic

```groovy
Auto-Approve IF ALL conditions true:
1. Prediction = PASS
2. Confidence ≥ 80%
3. Disk Space ≥ 100GB

Otherwise: Manual Approval Required
```

**Example scenarios:**

| Prediction | Confidence | Disk | Result |
|------------|------------|------|--------|
| PASS | 85% | 110GB | ✅ Auto-Approved |
| PASS | 75% | 110GB | ⚠️ Manual (low confidence) |
| PASS | 85% | 90GB | ⚠️ Manual (low disk) |
| FAIL | 85% | 110GB | ⚠️ Manual (failed prediction) |
| HIGH-RISK | 90% | 110GB | ⚠️ Manual (high-risk) |

***

## Feedback System

### Why Feedback Matters

The AI system **learns from your feedback** to improve predictions. The more feedback provided, the more accurate future predictions become.

### Feedback Flow

```
Build Completes
    ↓
AI predicted: HIGH-RISK @ 70%
Actual result: SUCCESS
    ↓
User provides feedback:
  - Actual: SUCCESS
  - Corrected confidence: 85%
  - Missed: "Build was actually fine"
  - False positives: "systemd version warning"
    ↓
Feedback sent to RAG API /api/feedback/submit
    ↓
AI System processes feedback:
  1. Marks prediction as incorrect
  2. Learns: systemd version warnings are false positives
  3. Adjusts confidence calculation for similar cases
  4. Updates rule performance metrics
    ↓
Next similar pipeline gets better prediction
```

### What to Include in Feedback

#### 1. **Actual Result**
Always provide if different from prediction:
```
AI Predicted: FAIL
Actual: SUCCESS ← Provide this!
```

#### 2. **Corrected Confidence**
Your assessment of the correct confidence:
```
AI said: 70% confidence
Your opinion: Should be 90% ← Provide this!
```

#### 3. **Missed Issues**
Problems AI didn't catch:
```
Build failed but AI said PASS because:
- Missing TMPDIR cleanup step
- Insufficient memory allocation
```

#### 4. **False Positives**
AI warned about non-issues:
```
AI warned about:
- systemd version conflict (actually compatible)
- Missing dependency (already inherited)
```

#### 5. **Comments**
Any additional context:
```
"Build succeeded but took longer due to network issues.
AI couldn't predict external factors."
```

### Feedback Types

**Automatic Feedback:**
- Sent when build completes without user input
- Includes: prediction ID, actual result, decision type
- Used for basic learning

**Manual Feedback:**
- Includes user corrections and explanations
- More valuable for AI learning
- Recommended for incorrect predictions

***

## Troubleshooting

### Common Issues

#### 1. API Connection Failed

**Symptom:**
```
ERROR: API failed: HTTP 000
```

**Causes & Solutions:**

```bash
# Check if RAG API is running
curl http://localhost:8000/health

# If not running, start it:
cd /home/azureuser/rag-system/modular_code_base
source ../rag_env/bin/activate
python -m app.main

# Check firewall
sudo ufw status
sudo ufw allow 8000/tcp

# Test from Jenkins server
curl http://your-api-server:8000/health
```

#### 2. No Pipelines Found

**Symptom:**
```
ERROR: No pipelines found
```

**Solution:**

```groovy
// Verify Jenkins has pipeline jobs
// Check Jenkins home for jobs:
ls -la /var/lib/jenkins/jobs/

// Ensure jobs are WorkflowJob type (Pipeline jobs)
// Script-based pipelines only, not Multibranch
```

#### 3. Timeout Waiting for Input

**Symptom:**
```
✗ Timeout/Abort
ABORTED
```

**Solution:**

```groovy
// Increase timeout in Approve stage
timeout(time: 30, unit: 'MINUTES') {  // Was 15
    // ...
}

// Or configure global timeout
options {
    timeout(time: 12, unit: 'HOURS')  // Was 8
}
```

#### 4. Disk Check Returns Wrong Value

**Symptom:**
```
Disk: 50GB (but actually have 200GB free)
```

**Solution:**

```groovy
// Check if correct partition is being checked
// Verify BUILD_DIR detection
echo "Build dir: ${env.BUILD_DIR}"

// Manually set if detection fails:
env.BUILD_DIR = '/home/jenkins/yocto-builds'

// Or modify findBuildDir() regex patterns
```

#### 5. Feedback Not Saving

**Symptom:**
```
✗ Failed (HTTP 500)
```

**Check RAG API logs:**
```bash
# View API logs
tail -f /home/azureuser/rag-system/modular_code_base/app.log

# Check database connection
psql -U rag_user -d yocto_analyzer

# Verify prediction_id exists
SELECT * FROM predictions WHERE id = 'your-uuid';
```

***

## Best Practices

### 1. Regular Feedback

**Recommendation:** Provide feedback on at least 50% of builds

```
Good practice:
- Always provide feedback when prediction is wrong
- Provide feedback on edge cases
- Add detailed comments for complex scenarios

Skip feedback when:
- Prediction and result match perfectly
- Very straightforward PASS @ 95%+ confidence
```

### 2. Threshold Tuning

**Start conservative, adjust based on accuracy:**

```
Week 1-2: AUTO_THRESHOLD = '85'  // Conservative
Week 3-4: AUTO_THRESHOLD = '80'  // Balanced (after AI learns)
Week 5+:  AUTO_THRESHOLD = '75'  // Aggressive (if accuracy >90%)
```

### 3. Resource Management

**Monitor disk space trends:**

```bash
# Check historical disk usage
df -h /yocto-builds

# Set alerts
if [ $(df --output=avail /yocto-builds | tail -1) -lt 70000000 ]; then
    echo "WARNING: Disk space low"
fi
```

### 4. Pipeline Organization

**Name pipelines descriptively:**

```
Good names:
- Yocto-Core-Poky-Zeus
- Custom-Recipe-Dunfell
- Integration-Test-Build

Bad names:
- Pipeline1
- Test
- Build
```

### 5. Analyze Before Major Changes

**Always run prediction pipeline:**
- After updating recipe versions
- Before merging branches
- After adding new dependencies
- When changing build configuration

***

## Advanced Configuration

### Custom Prediction Criteria

Modify auto-approval logic for your needs:

```groovy
// Example: Auto-approve only PASS with 90%+
def autoApprove = env.PRED == 'PASS' && 
                  conf >= 90 && 
                  env.DISK_OK == 'true'

// Example: Never auto-approve HIGH-RISK
def autoApprove = env.PRED == 'PASS' && 
                  conf >= env.AUTO_THRESHOLD.toInteger() && 
                  env.DISK_OK == 'true' &&
                  env.PRED != 'HIGH-RISK'

// Example: Add time-based rules (no auto on weekends)
def isWeekend = new Date().format('E') in ['Sat', 'Sun']
def autoApprove = !isWeekend && 
                  env.PRED == 'PASS' && 
                  conf >= 80 && 
                  env.DISK_OK == 'true'
```

### Custom Disk Patterns

If your build directories follow different naming:

```groovy
@NonCPS
def findBuildDir(text) {
    def patterns = [
        '/yocto-builds?[^\\s\'"]*',
        '/poky[^\\s\'"]*',
        '/my-custom-builds[^\\s\'"]*',  // Add your pattern
        'BUILDDIR[\\s=]+[\'"]?([^\\s\'"]+)',
        'BUILD_DIR[\\s=]+[\'"]?([^\\s\'"]+)'
    ]
    patterns.collectMany { p -> 
        (text =~ p).collect { it instanceof String ? it : it[1] } 
    }.unique().findAll { it?.size() > 3 }?.getAt(0) ?: '/tmp'
}
```

### Integration with Notifications

Add Slack/Email notifications:

```groovy
post {
    always {
        script {
            if (env.PRED == 'FAIL' && env.DECISION == 'AUTO') {
                slackSend(
                    color: 'danger',
                    message: "AI blocked build: ${env.TARGET} - ${env.PRED}@${env.CONF}%"
                )
            }
        }
    }
}
```

***

## Monitoring & Analytics

### View AI Learning Progress

```bash
# Check feedback statistics
curl http://localhost:8000/api/feedback/stats

# View learned patterns
curl http://localhost:8000/api/learning/patterns

# Check rule performance
curl http://localhost:8000/api/learning/rules

# View accuracy trend
curl http://localhost:8000/api/learning/accuracy-trend
```

### Jenkins Build History Analysis

```groovy
// Add to pipeline for tracking
archiveArtifacts artifacts: 'analysis.txt'

// View all predictions
http://jenkins:8080/job/AI-Yocto-Predictor/builds
```

***

## Summary

This pipeline provides an **intelligent gate** for Yocto builds by:

1. **Analyzing** build scripts with AI before execution
2. **Predicting** success/failure with confidence scores
3. **Auto-approving** safe builds to save time
4. **Learning** from feedback to improve accuracy
5. **Preventing** wasted resources on doomed builds

### Key Concepts

- **RAG (Retrieval Augmented Generation)**: AI retrieves relevant knowledge before answering
- **Confidence Score**: AI's certainty about prediction (0-100%)
- **Auto-Approval**: Automatic execution of high-confidence PASS predictions
- **Feedback Loop**: User corrections that improve AI accuracy



**You're now ready to use AI-powered build prediction!**