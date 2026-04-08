pipeline {
    agent any

    environment {
        DATA_PATH = "data.csv"
        OUTPUT_DIR = "outputs"
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Cloning repository...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                sh 'pip install --quiet -r requirements.txt'
            }
        }

        stage('Basic Dataset Tests') {
            steps {
                echo 'Running dataset sanity checks...'
                sh 'DATA_PATH=${DATA_PATH} pytest tests/test_dataset.py -v --tb=short'
            }
        }

        stage('Run ML Pipeline') {
            steps {
                echo 'Executing main pipeline...'
                sh 'python main.py --data ${DATA_PATH} --output ${OUTPUT_DIR}'
            }
        }

        stage('Archive Artifacts') {
            steps {
                echo 'Archiving metrics, plots and logs...'
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Pipeline finished successfully.'
        }
        failure {
            echo 'Pipeline failed. Check the logs above.'
        }
    }
}
