/**
 * RADIUS AI - INFERENCE ENGINE v10.0
 * HACKATHON EDITION
 */

const engine = {
    config: {
        apiBase: "http://127.0.0.1:8000"
    },

    state: {
        isProcessing: false,
        chart: null
    },

    init() {
        console.log("RADIUS_AI INFERENCE ENGINE v10.0 READY");
        this.setupEventListeners();
        
        if (window.pdfjsLib) {
            window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        }
    },

    setupEventListeners() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.onclick = () => fileInput.click();
        
        fileInput.onchange = (e) => {
            const file = e.target.files[0];
            if (file) this.handleFile(file);
        };

        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.background = '#0f172a'; dropZone.style.color = '#fff'; };
        dropZone.ondragleave = () => { dropZone.style.background = 'transparent'; dropZone.style.color = '#64748b';};
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.style.background = 'transparent';
            dropZone.style.color = '#0f172a';
            const file = e.dataTransfer.files[0];
            if (file) this.handleFile(file);
        };
    },

    async handleFile(file) {
        const fileLabel = document.getElementById('file-label');
        if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
            fileLabel.innerText = "PARSING PDF INGESTION...";
            try {
                const text = await this.extractTextFromPDF(file);
                document.getElementById('findings-text').value = text;
                fileLabel.innerText = `[SUCCESS] ${file.name}`;
                fileLabel.style.color = '#2563eb';
            } catch (err) {
                alert("INGESTION ERROR: Text-based PDF required.");
            }
        } else if (file.type === "text/plain" || file.name.endsWith(".txt")) {
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('findings-text').value = e.target.result;
                fileLabel.innerText = `[SUCCESS] ${file.name}`;
                fileLabel.style.color = '#2563eb';
            };
            reader.readAsText(file);
        }
    },

    async extractTextFromPDF(file) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        let fullText = "";
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            fullText += content.items.map(item => item.str).join(" ") + "\n";
        }
        return fullText.trim();
    },

    async execute() {
        if (this.state.isProcessing) return;

        const findings = document.getElementById('findings-text').value.trim();
        if (!findings) {
            alert("FATAL: No payload detected in DATA INGESTION zone.");
            return;
        }

        const data = {
            findings: findings,
            patient_name: document.getElementById('patient-name').value.trim() || "TARGET-UNKNOWN",
            patient_dob: document.getElementById('patient-dob').value || "UNKNOWN",
            patient_id: document.getElementById('patient-id').value.trim() || "REF-NULL",
            hospital_name: "RADIUS_AI LOCAL_NODE",
            referring_physician: "SYS_ADMIN"
        };

        this.setLoading(true);

        try {
            const response = await fetch(`${this.config.apiBase}/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error("API Node failure");

            const report = await response.json();
            this.render(report);

        } catch (error) {
            console.error(error);
            alert("EXECUTION FAILURE: " + error.message);
        } finally {
            this.setLoading(false);
        }
    },

    setLoading(active) {
        this.state.isProcessing = active;
        const btn = document.querySelector('.btn-primary');
        btn.innerText = active ? "PROCESSING..." : "EXECUTE INFERENCE";
        btn.style.opacity = active ? "0.5" : "1";
    },

    render(report) {
        document.getElementById('welcome-view').style.display = 'none';
        document.getElementById('report-view').style.display = 'block';

        // Headers
        document.getElementById('rep-exec-id').innerText = report.execution_id;
        document.getElementById('rep-timestamp').innerText = new Date(report.timestamp).toISOString();

        // Urgency logic
        const banner = document.getElementById('urgency-banner');
        if (report.severity.includes("HIGH RISK")) {
            banner.innerText = "CRITICAL: " + report.severity;
            banner.className = "urgency-banner urgency-high";
        } else {
            banner.innerText = "STATUS: " + report.severity;
            banner.className = "urgency-banner";
            banner.style.display = "block";
        }

        // Meta
        document.getElementById('rep-name').innerText = report.patient_name;
        document.getElementById('rep-dob').innerText = report.patient_dob;
        document.getElementById('rep-id').innerText = report.patient_id;
        document.getElementById('rep-technique').innerText = report.technique;

        // Diagnosis
        document.getElementById('rep-diagnosis').innerText = report.diagnosis.toUpperCase();
        document.getElementById('rep-area').innerText = report.affected_area.toUpperCase();
        document.getElementById('rep-latency').innerText = `LATENCY: ${report.latency_ms}ms`;

        // Actionable Directives
        document.getElementById('rep-recommendations').innerHTML = report.recommendations.map(r => `<li>${r}</li>`).join('');

        // Strict Horizontal Bar Chart
        this.renderChart(report.probability_distribution);
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
    },

    renderChart(distribution) {
        const ctx = document.getElementById('reliability-chart').getContext('2d');
        if (this.state.chart) this.state.chart.destroy();
        
        const labels = Object.keys(distribution).map(l => l.toUpperCase());
        const data = Object.values(distribution);

        // Map values accurately to percentages
        const percentageData = data.map(v => (v * 100).toFixed(1));

        this.state.chart = new Chart(ctx, {
            type: 'bar', // Horizontal Bar Chart
            data: {
                labels: labels,
                datasets: [{
                    data: percentageData,
                    backgroundColor: labels.map(l => l === 'NORMAL' ? '#0f172a' : '#ef4444'),
                    barThickness: 24,
                }]
            },
            options: {
                indexAxis: 'y', // This makes it horizontal
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false,
                        max: 100
                    },
                    y: {
                        grid: { display: false, drawBorder: false },
                        ticks: {
                            font: { family: "'IBM Plex Mono', monospace", size: 11, weight: '700' },
                            color: '#0f172a'
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.raw + '%';
                            }
                        },
                        titleFont: { family: "'IBM Plex Mono', monospace" },
                        bodyFont: { family: "'IBM Plex Mono', monospace" },
                        backgroundColor: '#0f172a',
                        cornerRadius: 0
                    }
                }
            }
        });
    }
};

document.addEventListener('DOMContentLoaded', () => engine.init());
