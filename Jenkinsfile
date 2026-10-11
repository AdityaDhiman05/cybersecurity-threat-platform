pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds(abortPrevious: true)
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    parameters {
        booleanParam(name: 'DEPLOY_TO_ECS', defaultValue: false,
                     description: 'Force a new ECS deployment after pushing the image')
    }

    environment {
        // ---- CHANGE THESE ----
        AWS_REGION     = 'ap-south-1'
        AWS_ACCOUNT_ID = '279391565053'
        S3_BUCKET      = 'threat-platform-artifacts-279391565053'
        ECS_CLUSTER    = 'threat-platform-cluster'
        ECS_SERVICE    = 'threat-platform'
        // Apple Silicon builds arm64 natively; taskdef.json must say "ARM64".
        PLATFORM       = 'linux/arm64'
        // -----------------------

        ECR_REPO     = 'cybersecurity-threat-platform'
        ECR_REGISTRY = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
        IMAGE_URI    = "${ECR_REGISTRY}/${ECR_REPO}"

        // Jenkins credential (kind: Username with password)
        //   username = AWS access key ID, password = AWS secret access key
        AWS_CREDS             = credentials('aws-ecr-creds')
        AWS_ACCESS_KEY_ID     = "${AWS_CREDS_USR}"
        AWS_SECRET_ACCESS_KEY = "${AWS_CREDS_PSW}"
        AWS_DEFAULT_REGION    = "${AWS_REGION}"

        // macOS specifics
        PATH            = "/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:${env.PATH}"
        DOCKER_HOST     = "unix://${env.HOME}/.docker/run/docker.sock"
        DOCKER_CONFIG   = "${env.WORKSPACE}/.docker"
        DOCKER_BUILDKIT = '1'
    }

    stages {
        stage('Preflight') {
            steps {
                sh '''
                    set -e
                    mkdir -p "$DOCKER_CONFIG"
                    docker version --format 'docker server {{.Server.Version}}'
                    aws sts get-caller-identity --query Arn --output text
                '''
            }
        }

        stage('Prepare') {
            steps {
                script {
                    env.GIT_SHA   = sh(returnStdout: true, script: 'git rev-parse --short=8 HEAD').trim()
                    env.IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_SHA}"
                }
                echo "Image tag: ${env.IMAGE_TAG}"
            }
        }

        stage('Fetch models + dataset from S3') {
            steps {
                sh '''
                    set -e
                    mkdir -p models data/raw
                    aws s3 sync s3://${S3_BUCKET}/models ./models
                    aws s3 cp   s3://${S3_BUCKET}/data/UNSW_NB15_testing-set.csv ./data/raw/UNSW_NB15_testing-set.csv
                '''
            }
        }

        stage('Verify models') {
            steps {
                sh '''
                    set -e
                    count=0
                    for f in models/*.pkl; do
                        size=$(wc -c < "$f" | tr -d ' ')
                        if [ "$size" -lt 1024 ]; then
                            echo "ERROR: $f is only $size bytes (pointer file, not the real model)"
                            exit 1
                        fi
                        count=$((count+1))
                    done
                    [ "$count" -eq 6 ] || { echo "ERROR: expected 6 model files, found $count"; exit 1; }
                    echo "All 6 model files look real."
                '''
            }
        }

        stage('Build image') {
            steps {
                sh '''
                    docker build --pull --platform ${PLATFORM} \
                      -t ${IMAGE_URI}:${IMAGE_TAG} \
                      -t ${IMAGE_URI}:latest .
                '''
            }
        }

        stage('Push to ECR') {
            steps {
                sh '''
                    set -e
                    aws ecr get-login-password --region ${AWS_REGION} \
                      | docker login --username AWS --password-stdin ${ECR_REGISTRY}
                    docker push ${IMAGE_URI}:${IMAGE_TAG}
                    docker push ${IMAGE_URI}:latest
                '''
            }
        }

        stage('Deploy to ECS Fargate') {
            when { expression { return params.DEPLOY_TO_ECS } }
            steps {
                sh '''
                    set -e
                    aws ecs update-service --cluster ${ECS_CLUSTER} --service ${ECS_SERVICE} --force-new-deployment
                    aws ecs wait services-stable --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE}
                '''
            }
        }
    }

    post {
        success { echo "Pushed ${env.IMAGE_URI}:${env.IMAGE_TAG}" }
        failure { echo 'Pipeline failed, check the stage logs above.' }
        always {
            sh '''
                docker rmi ${IMAGE_URI}:${IMAGE_TAG} ${IMAGE_URI}:latest || true
                docker logout ${ECR_REGISTRY} || true
                rm -rf "$DOCKER_CONFIG" || true
            '''
        }
    }
}