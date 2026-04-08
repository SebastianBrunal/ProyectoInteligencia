pipeline {
    // ── Corre todo dentro de un contenedor Python limpio ──────────────────
    agent {
        docker {
            image 'python:3.11-slim'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }

    environment {
        DATA_PATH  = "data.csv"
        OUTPUT_DIR = "outputs"
    }

    stages {

        // ── 1. Checkout ───────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                echo 'Clonando repositorio...'
                checkout scm
            }
        }

        // ── 2. Dependencias ───────────────────────────────────────────────
        stage('Install Dependencies') {
            steps {
                echo 'Instalando dependencias Python...'
                sh 'pip install --quiet --no-cache-dir -r requirements.txt'
            }
        }

        // ── 3. Tests del dataset ──────────────────────────────────────────
        stage('Dataset Tests') {
            steps {
                echo 'Corriendo checks del dataset SDSS...'
                sh 'DATA_PATH=${DATA_PATH} pytest tests/test_dataset.py -v --tb=short'
            }
        }

        // ── 4. Pipeline ML ────────────────────────────────────────────────
        stage('Run ML Pipeline') {
            steps {
                echo 'Ejecutando pipeline de ML...'
                sh 'python main.py --data ${DATA_PATH} --output ${OUTPUT_DIR}'
            }
        }

        // ── 5. Publicar artefactos ────────────────────────────────────────
        stage('Archive Artifacts') {
            steps {
                echo 'Archivando métricas y gráficas...'
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline finalizado exitosamente.'
        }
        failure {
            echo '❌ Pipeline falló. Revisar logs arriba.'
        }
        always {
            // Limpia el workspace después de cada build
            cleanWs()
        }
    }
}
