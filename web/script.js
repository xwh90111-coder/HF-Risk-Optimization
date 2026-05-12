document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictionOutput = document.getElementById('prediction-output');
    const predictButton = document.getElementById('predict-button');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        // --- 1. Visual Feedback: Loading State --- 
        predictButton.disabled = true;
        predictButton.textContent = '分析数据中...';
        resultContainer.classList.add('hidden');

        // --- 2. Collect and Validate Form Data --- 
        const formData = new FormData(form);
        const payload = {};
        let formIsValid = true;
        for (let [key, value] of formData.entries()) {
            if (value === '' || value === null) {
                formIsValid = false;
                break;
            }
            payload[key] = Number(value);
        }

        if (!formIsValid) {
            displayError('请填写所有字段后再进行预测。');
            predictButton.disabled = false;
            predictButton.textContent = '立即预测';
            return;
        }
        
        // For debugging: Log the data being sent to the server
        console.log('Sending payload:', JSON.stringify(payload));

        // --- 3. API Call with Error Handling --- 
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            // Check if the server responded with an error code
            if (!response.ok) {
                const errorData = await response.json().catch(() => null); // Try to get JSON error, otherwise null
                const errorMessage = errorData?.error || `服务器响应错误: ${response.status}`;
                throw new Error(errorMessage);
            }

            const result = await response.json();
            
            // For debugging: Log the data received from the server
            console.log('Received result:', result);

            displayResult(result);

        } catch (error) {
            console.error('Prediction failed:', error);
            displayError(error.message || '无法连接到服务器或发生未知错误。');
        } finally {
            // --- 4. Restore Button State --- 
            predictButton.disabled = false;
            predictButton.textContent = '立即预测';
        }
    });

    function displayResult(result) {
        predictionOutput.innerHTML = ''; // Clear previous results

        const resultCard = document.createElement('div');
        resultCard.className = result.prediction === 1 ? 'result-card risk' : 'result-card safe';

        const predictionText = document.createElement('p');
        predictionText.className = 'prediction-text';
        predictionText.textContent = result.prediction === 1 ? '高风险' : '低风险';

        const probabilityText = document.createElement('p');
        probabilityText.className = 'probability';
        probabilityText.textContent = `心力衰竭可能性: ${(result.probability * 100).toFixed(1)}%`;

        resultCard.append(predictionText, probabilityText);
        predictionOutput.appendChild(resultCard);

        if (result.risk_factors && result.risk_factors.length > 0) {
            const riskFactorsContainer = document.createElement('div');
            riskFactorsContainer.className = 'risk-factors';
            riskFactorsContainer.innerHTML = `<h3>检测到的主要风险因素</h3>`;
            
            const list = document.createElement('ul');
            result.risk_factors.forEach(factor => {
                const item = document.createElement('li');
                item.textContent = factor;
                list.appendChild(item);
            });
            riskFactorsContainer.appendChild(list);
            predictionOutput.appendChild(riskFactorsContainer);
        }

        resultContainer.classList.remove('hidden');
        // Scroll to results for better user experience
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function displayError(message) {
        predictionOutput.innerHTML = `
            <div class="result-card risk">
                <p class="prediction-text">预测出错</p>
                <p class="probability" style="font-size: 1rem;">${message}</p>
            </div>
        `;
        resultContainer.classList.remove('hidden');
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});