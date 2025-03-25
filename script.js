function checkNews() {
    let newsText = document.getElementById("newsInput").value;
    let result = document.getElementById("result");
    let resultContainer = document.getElementById("result-container");

    if (newsText.trim() === "") {
        result.innerHTML = "⚠️ Please enter some text!";
        result.style.color = "black";
        resultContainer.style.backgroundColor = "#f4f4f4";
        resultContainer.style.display = "block";
        return;
    }

    // Send the news text to Flask API
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ news: newsText }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction === "Fake News") {
            result.innerHTML = "❌ Fake News Detected!";
            resultContainer.style.backgroundColor = "red";
        } else {
            result.innerHTML = "✔️ This News Looks Legit!";
            resultContainer.style.backgroundColor = "green";
        }
        resultContainer.style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        result.innerHTML = "⚠️ Error in prediction!";
        resultContainer.style.backgroundColor = "gray";
        resultContainer.style.display = "block";
    });
}