pipeline {
    agent none

    environment {
        DATA_PATH  = "data.csv"
        OUTPUT_DIR = "outputs"
        IMAGE_NAME = "sdss-ml"
    }

    stages {

        stage('Checkout') {
            agent { docker { image 'python:3.11-slim' args '-v /var/run/docker.sock:/var/run/docker.sock' } }
            steps {
                echo 'Clonando repositorio...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            agent { docker { image 'python:3.11-slim' args '-v /var/run/docker.sock:/var/run/docker.sock' } }
            steps {
                echo 'Instalando dependencias Python...'
                sh 'pip install --quiet --no-cache-dir -r requirements.txt'
            }
        }

        stage('Dataset Tests') {
            agent { docker { image 'python:3.11-slim' args '-v /var/run/docker.sock:/var/run/docker.sock' } }
            steps {
                echo 'Corriendo checks del dataset SDSS...'
                sh 'DATA_PATH=${DATA_PATH} pytest tests/test_dataset.py -v --tb=short'
            }
        }

        stage('Run ML Pipeline') {
            agent { docker { image 'python:3.11-slim' args '-v /var/run/docker.sock:/var/run/docker.sock' } }
            steps {
                echo 'Ejecutando pipeline de ML...'
                sh 'pip install --quiet --no-cache-dir -r requirements.txt && python main.py --data ${DATA_PATH} --output ${OUTPUT_DIR}'
            }
        }

        stage('Build Docker Image') {
            agent any
            steps {
                echo 'Construyendo imagen Docker con Streamlit...'
                checkout scm
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }

        stage('Deploy Streamlit App') {
            agent any
            steps {
                echo 'Lanzando app Streamlit...'
                sh '''
                    docker stop sdss-app || true
                    docker rm sdss-app || true
                    docker run -d \
                        --name sdss-app \
                        -p 8501:8501 \
                        ${IMAGE_NAME}
                '''
                echo 'App disponible en http://localhost:8501'
            }
        }

        stage('Archive Artifacts') {
            agent { docker { image 'python:3.11-slim' args '-v /var/run/docker.sock:/var/run/docker.sock' } }
            steps {
                echo 'Archivando métricas y gráficas...'
                sh 'pip install --quiet --no-cache-dir -r requirements.txt && python main.py --data ${DATA_PATH} --output ${OUTPUT_DIR}'
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline finalizado. App corriendo en http://localhost:8501'
        }
        failure {
            echo '❌ Pipeline falló. Revisar logs arriba.'
        }
    }
}