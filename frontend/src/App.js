import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  React.useEffect(() => {
    // Fetch model info on load
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/model-info');
      setModelInfo(response.data);
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!uploadedFile) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await axios.post('http://localhost:8000/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
    } catch (err) {
      setError('Error making prediction: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (confidence) => {
    if (confidence >= 80) return '#16a34a'; // green
    if (confidence >= 60) return '#f39c12'; // orange
    return '#e74c3c'; // red
  };

  const getSeverityLabel = (confidence) => {
    if (confidence >= 80) return 'High';
    if (confidence >= 60) return 'Moderate';
    return 'Low';
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="title">🍃 Plant Disease Detection</h1>
          <p className="subtitle">AI-Powered Disease Detection for Plant Leaves</p>
        </div>
      </header>

      <div className="container">
        <div className="sidebar">
          <div className="sidebar-section">
            <h3>📊 Model Information</h3>
            {modelInfo ? (
              <div className="stats">
                <div className="stat">
                  <span className="stat-label">Training Accuracy</span>
                  <span className="stat-value">{modelInfo.train_acc}%</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Validation Accuracy</span>
                  <span className="stat-value">{modelInfo.val_acc}%</span>
                </div>
              </div>
            ) : (
              <p>Loading...</p>
            )}
          </div>
        </div>

        <div className="main-content">
          <div className="content-grid">
            {/* Upload Section */}
            <div className="card upload-card">
              <h2>📤 Upload Plant Image</h2>
              <div className="upload-area">
                <input
                  type="file"
                  id="file-input"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="file-input"
                />
                <label htmlFor="file-input" className="upload-label">
                  <span className="upload-icon">📁</span>
                  <span className="upload-text">
                    {uploadedFile ? uploadedFile.name : 'Click or drag image here'}
                  </span>
                  <span className="upload-hint">JPG, PNG, BMP, GIF up to 200MB</span>
                </label>
              </div>

              {preview && (
                <div className="preview-container">
                  <img src={preview} alt="preview" className="preview-image" />
                </div>
              )}

              <button
                onClick={handlePredict}
                disabled={!uploadedFile || loading}
                className="analyze-btn"
              >
                {loading ? '🔄 Analyzing...' : '🔮 Analyze Image'}
              </button>
            </div>

            {/* Results Section */}
            <div className="card results-card">
              <h2>🔮 Analysis Results</h2>

              {error && (
                <div className="error-message">
                  <p>⚠️ {error}</p>
                </div>
              )}

              {!results && !loading && !error && (
                <div className="empty-state">
                  <p className="empty-icon">🛡️</p>
                  <p className="empty-text">No Analysis Yet</p>
                  <p className="empty-hint">Upload an image to get started</p>
                </div>
              )}

              {results && (
                <div className="result-content">
                  <div className="main-result">
                    <h3 className="disease-name">{results.main_result.display}</h3>
                    <div className="confidence-container">
                      <span className="confidence-label">Confidence:</span>
                      <span
                        className="confidence-value"
                        style={{ color: getSeverityColor(results.main_result.confidence) }}
                      >
                        {results.main_result.confidence.toFixed(2)}%
                      </span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${results.main_result.confidence}%`,
                          backgroundColor: getSeverityColor(results.main_result.confidence),
                        }}
                      ></div>
                    </div>
                    <div className="severity-badge" style={{ backgroundColor: getSeverityColor(results.main_result.confidence) + '20', borderLeft: `4px solid ${getSeverityColor(results.main_result.confidence)}` }}>
                      <span>Severity: {getSeverityLabel(results.main_result.confidence)}</span>
                    </div>
                  </div>

                  <div className="treatment-section">
                    <h4>💊 Treatment Recommendation</h4>
                    <p>{results.main_result.remedy}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Top 3 Predictions */}
          {results && results.top_predictions && (
            <div className="predictions-section">
              <h2>📊 Top 3 Predictions</h2>
              <div className="predictions-grid">
                {results.top_predictions.map((pred, idx) => (
                  <div key={idx} className="prediction-box">
                    <div className="prediction-rank">{idx + 1}</div>
                    <h4>{pred.display}</h4>
                    <div className="prediction-confidence">{pred.confidence.toFixed(1)}%</div>
                    <div className="mini-progress">
                      <div
                        className="mini-progress-fill"
                        style={{
                          width: `${pred.confidence}%`,
                          backgroundColor: getSeverityColor(pred.confidence),
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Detections */}
          <div className="recent-section">
            <h2>📋 Recent Detections</h2>
            <div className="recent-grid">
              <div className="recent-item">
                <span className="recent-name">Tomato Leaf Blight</span>
                <span className="recent-confidence">92%</span>
                <span className="recent-severity" style={{ background: '#fee2e2', color: '#991b1b' }}>Moderate</span>
              </div>
              <div className="recent-item">
                <span className="recent-name">Rice Blast</span>
                <span className="recent-confidence">87%</span>
                <span className="recent-severity" style={{ background: '#fee2e2', color: '#991b1b' }}>High</span>
              </div>
              <div className="recent-item">
                <span className="recent-name">Wheat Rust</span>
                <span className="recent-confidence">95%</span>
                <span className="recent-severity" style={{ background: '#dcfce7', color: '#166534' }}>Low</span>
              </div>
              <div className="recent-item">
                <span className="recent-name">Potato Late Blight</span>
                <span className="recent-confidence">89%</span>
                <span className="recent-severity" style={{ background: '#fee2e2', color: '#991b1b' }}>High</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>🌾 Plant Disease Detection v2.0 | Powered by React & TensorFlow</p>
      </footer>
    </div>
  );
}

export default App;
