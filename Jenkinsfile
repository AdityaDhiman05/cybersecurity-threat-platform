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
    }

    stages {
        stage('Preflight') {
            steps {
                sh '''
                    set -e
                    mkdir -p "$DOCKER_CONFIG"
                    if [ -d "$HOME/.docker/cli-plugins" ]; then
                        ln -sfn "$HOME/.docker/cli-plugins" "$DOCKER_CONFIG/cli-plugins"
                    fi
                    docker version --format 'docker server {{.Server.Version}}'
                    docker buildx version || echo "buildx not available, build will use the classic builder"
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
                // The repo tracks models/*.pkl as pointer files, so every checkout resets them.
                // Keep the real files in a cache OUTSIDE the workspace, refresh it from S3
                // (only missing or changed files download), then copy them into the workspace.
                sh '''
                    set -e
                    CACHE="$HOME/.tp-artifact-cache"
                    mkdir -p "$CACHE/models" "$CACHE/data" models data/raw
                    export AWS_MAX_ATTEMPTS=10 AWS_RETRY_MODE=standard

                    retry() {
                        try=1
                        until "$@"; do
                            if [ "$try" -ge 4 ]; then
                                echo "ERROR: S3 download failed after $try attempts"
                                exit 1
                            fi
                            echo "S3 sync failed (attempt $try), retrying in 10s..."
                            try=$((try+1))
                            sleep 10
                        done
                    }

                    retry aws s3 sync "s3://${S3_BUCKET}/models" "$CACHE/models" --only-show-errors
                    retry aws s3 sync "s3://${S3_BUCKET}/data" "$CACHE/data" --exclude "*" --include "UNSW_NB15_testing-set.csv" --only-show-errors

                    if [ ! -s "$CACHE/data/UNSW_NB15_testing-set.csv" ]; then
                        echo "ERROR: dataset missing in S3. Upload it with:"
                        echo "  aws s3 cp UNSW_NB15_testing-set.csv s3://${S3_BUCKET}/data/UNSW_NB15_testing-set.csv"
                        exit 1
                    fi

                    for f in "$CACHE"/models/*.pkl; do
                        rm -f "models/$(basename "$f")"
                        cp -c "$f" models/ 2>/dev/null || cp "$f" models/
                    done
                    rm -f data/raw/UNSW_NB15_testing-set.csv
                    cp -c "$CACHE/data/UNSW_NB15_testing-set.csv" data/raw/ 2>/dev/null || cp "$CACHE/data/UNSW_NB15_testing-set.csv" data/raw/
                    ls -lh models/ data/raw/
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
                    if docker buildx version > /dev/null 2>&1; then
                        export DOCKER_BUILDKIT=1
                    else
                        export DOCKER_BUILDKIT=0
                        echo "buildx not available, using the classic builder"
                    fi
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