pipeline {
    agent any
 
    environment {
        DATA_PATH  = "data.csv"
        OUTPUT_DIR = "outputs"
        IMAGE_NAME = "sdss-ml"
    }
 
    stages {
 
        stage('Checkout') {
            steps {
                echo 'Clonando repositorio...'
                checkout scm
            }
        }
 
        stage('Install Dependencies') {
            steps {
                echo 'Instalando dependencias Python...'
                sh 'pip install --quiet --no-cache-dir -r requirements.txt'
            }
        }
 
        stage('Dataset Tests') {
            steps {
                echo 'Corriendo checks del dataset SDSS...'
                sh 'DATA_PATH=${DATA_PATH} pytest tests/test_dataset.py -v --tb=short'
            }
        }
 
        stage('Run ML Pipeline') {
            steps {
                echo 'Ejecutando pipeline de ML...'
                sh 'python main.py --data ${DATA_PATH} --output ${OUTPUT_DIR}'
            }
        }
 
        stage('Build Docker Image') {
            steps {
                echo 'Construyendo imagen Docker...'
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }
 
        stage('Deploy Streamlit App') {
            steps {
                echo 'Lanzando app Streamlit...'
                sh '''
                    docker stop sdss-app || true
                    docker rm sdss-app || true
                    docker run -d --name sdss-app -p 8501:8501 ${IMAGE_NAME}
                '''
                echo 'App disponible en http://localhost:8501'
            }
        }
 
        stage('Archive Artifacts') {
            steps {
                echo 'Archivando resultados...'
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
            }
        }
    }
 
    post {
        success {
            echo '✅ Pipeline finalizado. App en http://localhost:8501'
        }
        failure {
            echo '❌ Pipeline falló. Revisar logs.'
        }
    }
}
 
 