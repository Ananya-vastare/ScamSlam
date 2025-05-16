const express = require('express');
const path = require('path');
const app = express();
const fetch = require('node-fetch');  // Run: npm install node-fetch@2

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'HomePage.html'));
});

app.get("/MainHub", (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'MainHub.html'));
});

app.post("/MainHub/submit", async (req, res) => {
    try {
        const response = await fetch('http://localhost:5000/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body),
        });
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error("Error calling Flask API:", error);
        res.status(500).json({ error: "Internal server error" });
    }
});


app.listen(3000, () => {
    console.log("Server running at http://localhost:3000");
});
