window.onload = () => {
    fetch("/data")
    .then(res => res.json())
    .then(data => {

        if (!data || data.length === 0) return;

        const emotionIndex = {
            angry: 0,
            disgust: 1,
            fear: 2,
            sad: 3,
            neutral: 4,
            happy: 5,
            surprise: 6
        };

        // ✅ SAFE predict polling
        setInterval(() => {
            fetch("/predict")
            .then(res => res.json())
            .then(data => {
                const el = document.getElementById("prediction");
                if (el) {
                    el.innerText =
                        "Current: " + data.current + 
                        " → Next: " + data.predicted;
                }
            })
            .catch(() => {}); // silent fail
        }, 2000);

        const startTime = data[0].time;
        const times = data.map(d => (d.time - startTime).toFixed(2));

        const values = data.map(d => emotionIndex[d.emotion] ?? 4);

        const ctx = document.getElementById("chart");
        if (!ctx) return;

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: times,
                datasets: [{
                    label: 'Emotion Timeline',
                    data: values,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56,189,248,0.2)',
                    fill: false,
                    tension: 0.3
                }]
            },
            options: {
                scales: {
                    y: {
                        ticks: {
                            callback: function(value) {
                                return Object.keys(emotionIndex)
                                    .find(key => emotionIndex[key] === value);
                            }
                        },
                        min: 0,
                        max: 6
                    },
                    x: {
                        title: {
                            display: true,
                            text: "Time (seconds)"
                        }
                    }
                }
            }
        });
    });
};