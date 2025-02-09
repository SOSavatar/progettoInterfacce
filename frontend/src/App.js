import React, { useState } from "react";
import axios from "axios";
import "./App.css";


function App() {
  const [file, setFile] = useState(null);
  const [fileContent, setFileContent] = useState(""); // Stato per il contenuto del file
  const [graph, setGraph] = useState(""); // Stato per il grafico

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    if (file && !file.name.endsWith(".txt")) {
      alert("Please select a .txt file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert("File uploaded successfully: " + response.data.message);
      
      // Imposta il contenuto del file e il grafico nel frontend
      setFileContent(response.data.content);
      setGraph(response.data.graph);  // Imposta l'immagine Base64 nel grafico
    } catch (error) {
      console.error("Error uploading file", error);
      alert("Error uploading file");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="title">Emotion Analyzer</h1>
        <div className="upload-box">     
          <input type="file" accept=".txt" onChange={handleFileChange} className="file-input" />
          <button onClick={handleUpload} className="analyze-button">
            Analyze Emotions
          </button>
        </div>
        {fileContent && (
          <div className="file-content">
            <h2>Genere Predetto:</h2>
            <p>{fileContent}</p>
          </div>
        )}
        {graph && (  // Mostra il grafico solo se esiste
          <div className="graph-content">
            <h2>Emotion Graph:</h2>
            <img src={`data:image/png;base64,${graph}`} alt="Emotion Graph" className = "graph-image" />
          </div>
        )}
      </header>
    </div>
  );
}

export default App;

