document.getElementById("matchForm").addEventListener("submit", async function (e) {
    e.preventDefault();  // Prevent default form submission behavior

    const formData = new FormData(this);  // Collect form data

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const resultText = await response.text();  // Expect HTML response as text
        displayResult(resultText);  // Display the result in the result div

    } catch (error) {
        console.error("Error:", error);
    }
});

function displayResult(htmlContent) {
    document.getElementById("results").innerHTML = htmlContent;  // Insert HTML into the results div
}

function toggleForm() {
    const caseType = document.querySelector('input[name="case_type"]:checked').value;
    const jobDescField = document.getElementById('job_description_field');
    if (caseType === 'both') {
        jobDescField.style.display = 'block';  // Show job description field
    } else {
        jobDescField.style.display = 'none';  // Hide job description field
    }
}
