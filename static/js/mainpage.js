window.onload = function () {
    let selectedType = "";

    const dropbtn = document.querySelector('.dropbtn');
    const dropdownContent = document.querySelector('.dropdown-content');
    const items = document.querySelectorAll('.dropdown-content li');
    const form = document.getElementById('container1');
    const inputBox = document.getElementById('input-box');
    const fileInput = document.getElementById('file-input');  // might be null
    const resultDiv = document.getElementById('result');
    const alertSound = document.getElementById("alertsound");
    const successSound = document.getElementById("sucess");

    if (fileInput) {
        fileInput.style.display = 'none';
    }

    dropbtn.addEventListener('click', () => {
        dropdownContent.classList.toggle('open');
    });

    items.forEach(item => {
        item.addEventListener('click', () => {
            const selection = item.textContent.trim();

            selectedType = selection;

            if (selection === 'SMS') {
                inputBox.style.display = 'block';
                if (fileInput) fileInput.style.display = 'none';
                inputBox.placeholder = 'Enter the SMS content you have received';
            } else {
                inputBox.style.display = 'block';
                if (fileInput) fileInput.style.display = 'none';

                inputBox.placeholder =
                    selection === 'Email' ? 'Enter the email contents you have received' :
                        selection === 'URL' ? 'Enter the URL' : '';
            }

            form.style.display = 'flex';
            dropdownContent.classList.remove('open');
        });
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        resultDiv.textContent = "Processing...";

        if (!selectedType) {
            alert("Please select a type first.");
            return;
        }

        try {
            const inputValue = inputBox.value.trim();
            if (!inputValue) {
                alert("Please enter input.");
                return;
            }

            const response = await fetch('/MainHub/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: selectedType, input: inputValue })
            });

            inputBox.value = "";

            if (!response.ok) throw new Error(`Server responded with status ${response.status}`);

            const data = await response.json();

            if (data.result) {
                resultDiv.textContent = data.result;

                if (data.result.toLowerCase().includes("not")) {
                    successSound.currentTime = 0;
                    successSound.play().catch(e => console.log('Alert sound play failed:', e));
                } else {
                    alertSound.currentTime = 0;
                    alertSound.play().catch(e => console.log('Success sound play failed:', e));
                }
            } else if (data.error) {
                resultDiv.textContent = "Error: " + data.error;
            } else {
                resultDiv.textContent = "No valid response from server.";
            }

        } catch (error) {
            resultDiv.textContent = `Error: Could not connect to server. (${error.message})`;
        }
    });
};
