// ============================================
//  Traffic AI Dashboard - JavaScript
// ============================================

const API = 'http://localhost:5000';
let statsInterval = null;
let vehiclesInterval = null;

// ---- Stats Polling ----
function fetchStats() {
    fetch(API + '/stats')
        .then(r => r.json())
        .then(data => {
            animateValue('stat-total', data.total);
            animateValue('stat-car', data.car);
            animateValue('stat-bike', data.bike);
            animateValue('stat-bus', data.bus);
            animateValue('stat-truck', data.truck);

            // Mode badge
            document.getElementById('mode-badge').textContent = data.mode || 'SVM/MOG2';

            // Elapsed
            document.getElementById('elapsed').textContent = Math.round(data.elapsed_seconds || 0) + 's';

            // Density
            updateDensity(data.density || 'Low');

            // Chart
            drawChart(data.car, data.bike, data.bus, data.truck);
        })
        .catch(() => {
            document.getElementById('status-badge').textContent = '⚠ OFFLINE';
            document.getElementById('status-badge').className = 'badge';
        });
}

function animateValue(id, newVal) {
    const el = document.getElementById(id);
    const cur = parseInt(el.textContent) || 0;
    if (cur !== newVal) {
        el.textContent = newVal;
        el.style.transform = 'scale(1.15)';
        el.style.transition = 'transform 0.2s';
        setTimeout(() => { el.style.transform = 'scale(1)'; }, 200);
    }
}

function updateDensity(density) {
    const bar = document.getElementById('density-bar');
    const text = document.getElementById('density-text');
    text.textContent = density.toUpperCase();

    switch (density) {
        case 'High':
            bar.style.background = 'linear-gradient(135deg, #c62828, #ef5350)';
            bar.style.width = '100%';
            break;
        case 'Medium':
            bar.style.background = 'linear-gradient(135deg, #f57f17, #ffc107)';
            bar.style.width = '66%';
            break;
        default:
            bar.style.background = 'linear-gradient(135deg, #00c853, #69f0ae)';
            bar.style.width = '33%';
    }
}

// ---- Vehicles Table ----
function fetchVehicles() {
    fetch(API + '/vehicles')
        .then(r => r.json())
        .then(data => {
            const tbody = document.getElementById('vehicle-tbody');
            tbody.innerHTML = '';
            const recent = data.slice(0, 30);
            recent.forEach(v => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${v.vehicle_id}</td>
                    <td><span class="type-badge type-${v.type}">${v.type}</span></td>
                    <td>${v.speed ? v.speed.toFixed(1) : '—'} px/f</td>
                    <td>${v.lane || '—'}</td>
                    <td>${v.timestamp ? v.timestamp.split(' ')[1] || v.timestamp : '—'}</td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(() => {});
}

// ---- Chart Drawing (Canvas) ----
function drawChart(cars, bikes, buses, trucks) {
    const canvas = document.getElementById('chart-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const values = [cars, bikes, buses, trucks];
    const labels = ['Cars', 'Bikes', 'Buses', 'Trucks'];
    const colors = ['#00e676', '#ff4757', '#42a5f5', '#ffa726'];
    const max = Math.max(...values, 1);

    const barW = 60;
    const gap = 30;
    const startX = (w - (barW * 4 + gap * 3)) / 2;
    const chartH = h - 60;
    const baseY = h - 30;

    values.forEach((val, i) => {
        const x = startX + i * (barW + gap);
        const barH = (val / max) * chartH * 0.85;

        // Bar gradient
        const grad = ctx.createLinearGradient(x, baseY, x, baseY - barH);
        grad.addColorStop(0, colors[i]);
        grad.addColorStop(1, colors[i] + '66');

        // Shadow
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.beginPath();
        ctx.roundRect(x + 3, baseY - barH + 3, barW, barH, 6);
        ctx.fill();

        // Bar
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, baseY - barH, barW, barH, 6);
        ctx.fill();

        // Value on top
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(val, x + barW / 2, baseY - barH - 8);

        // Label below
        ctx.fillStyle = '#8890b0';
        ctx.font = '12px Inter';
        ctx.fillText(labels[i], x + barW / 2, baseY + 18);
    });
}

// ---- Controls ----
function setMode(mode) {
    fetch(API + '/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
    }).then(() => {
        document.getElementById('btn-svm').classList.toggle('active', mode === 'svm');
        document.getElementById('btn-yolo').classList.toggle('active', mode === 'yolo');
    }).catch(err => console.error('Mode switch failed:', err));
}

function resetCounters() {
    fetch(API + '/reset', { method: 'POST' })
        .then(() => {
            ['stat-total', 'stat-car', 'stat-bike', 'stat-bus', 'stat-truck']
                .forEach(id => { document.getElementById(id).textContent = '0'; });
        })
        .catch(err => console.error('Reset failed:', err));
}

function uploadVideo(input) {
    const file = input.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('video', file);
    fetch(API + '/set_video', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'success') {
                // Refresh video feed
                const img = document.getElementById('video-feed');
                img.src = API + '/video_feed?' + Date.now();
            }
        })
        .catch(err => console.error('Upload failed:', err));
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchVehicles();
    statsInterval = setInterval(fetchStats, 1000);
    vehiclesInterval = setInterval(fetchVehicles, 3000);

    // Canvas roundRect polyfill
    if (!CanvasRenderingContext2D.prototype.roundRect) {
        CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
            if (w < 2 * r) r = w / 2;
            if (h < 2 * r) r = h / 2;
            this.moveTo(x + r, y);
            this.arcTo(x + w, y, x + w, y + h, r);
            this.arcTo(x + w, y + h, x, y + h, r);
            this.arcTo(x, y + h, x, y, r);
            this.arcTo(x, y, x + w, y, r);
            this.closePath();
            return this;
        };
    }
});
