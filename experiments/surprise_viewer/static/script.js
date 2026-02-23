document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const inputText = document.getElementById('inputText');
    const heatmapContainer = document.getElementById('heatmapContainer');
    const status = document.getElementById('status');

    analyzeBtn.addEventListener('click', async () => {
        const text = inputText.value;
        if (!text) return;

        // Reset UI
        heatmapContainer.innerHTML = '';
        analyzeBtn.disabled = true;
        status.textContent = 'Analyzing...';

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                // Process all complete lines
                buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        appendToken(data);
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }
                }
            }

            status.textContent = 'Analysis complete.';
        } catch (error) {
            console.error('Error:', error);
            status.textContent = 'Error occurred.';
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function appendToken(data) {
        const span = document.createElement('span');
        // Handle whitespace in tokens for proper rendering
        const content = data.token;
        span.textContent = content;
        span.className = 'token';

        // Probability Mismatch: 1.0 - probability
        const probability = data.probability !== undefined ? data.probability : 1.0;
        const mismatch = 1.0 - probability;

        // Map mismatch (0 to 1) to Hue (Green to Red)
        // Green: 120, Red: 0
        const hue = 120 - (mismatch * 120);

        // Background color with mismatch-based opacity
        // Higher mismatch -> Darker/Stronger Red
        // Lower mismatch -> Darker/Stronger Green
        // We use a constant lightness/saturation for "rectangular highlights"
        const lightness = 40;
        const saturation = 80;
        const opacity = 0.6 + (mismatch * 0.3); // 0.6 to 0.9

        span.style.backgroundColor = `hsla(${hue}, ${saturation}%, ${lightness}%, ${opacity})`;

        // Tooltip
        span.title = `Surprise: ${data.surprise.toFixed(2)}\nProbability: ${probability.toFixed(4)}\nMismatch: ${mismatch.toFixed(4)}`;

        heatmapContainer.appendChild(span);

        // Ensure whitespace is preserved
        if (content.match(/^\s/)) {
            // If it starts with space, browsers might collapse it if not careful
            // But individual spans usually handle it OK if display: inline-block
        }
    }
});
