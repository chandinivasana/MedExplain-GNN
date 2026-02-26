'use client';

import { useState, useEffect } from 'react';

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<any[]>([]);

  const fetchHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/history');
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict-disease', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
      
      // Refresh history slightly after prediction to ensure DB write completion
      setTimeout(fetchHistory, 500);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8 bg-gray-50 flex flex-col md:flex-row items-start justify-center gap-8">
      
      {/* Left Column: Input and Current Result */}
      <div className="w-full max-w-2xl flex flex-col items-center">
        <h1 className="text-4xl font-bold mb-8 text-blue-900 self-start">MedExplain-GNN</h1>
        <form onSubmit={handleSubmit} className="w-full bg-white p-6 rounded-lg shadow-md mb-8">
          <label className="block text-gray-700 font-semibold mb-2">Describe your symptoms:</label>
          <textarea
            className="w-full p-3 border rounded-md mb-4 focus:ring-2 focus:ring-blue-500 text-black"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. I have a high fever and joint pain since yesterday..."
          ></textarea>
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded transition duration-200"
          >
            {loading ? 'Processing...' : 'Analyze Symptoms'}
          </button>
        </form>

        {result && (
          <div className="w-full bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-2 text-red-600">Prediction: {result.disease}</h2>
            <p className="text-gray-600 mb-4">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            <div className="mb-4">
              <h3 className="font-semibold text-gray-800">Reasoning:</h3>
              <p className="text-gray-700 italic">{result.explanation}</p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-800">Foods to Avoid:</h3>
              <ul className="list-disc list-inside text-red-700">
                {result.dietary_precautions.map((food: string, i: number) => (
                  <li key={i}>{food}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* Right Column: History */}
      <div className="w-full max-w-md bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4 text-gray-800 border-b pb-2">Session History</h2>
        {history.length === 0 ? (
          <p className="text-gray-500 italic">No previous predictions found.</p>
        ) : (
          <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
            {history.map((item, index) => (
              <div key={item._id || index} className="p-4 border rounded-md bg-gray-50 flex flex-col gap-2">
                <p className="text-sm text-gray-800 italic line-clamp-2">"{item.input_text}"</p>
                <div className="flex justify-between items-center">
                  <span className="font-bold text-red-600">{item.prediction.disease}</span>
                  <span className="text-xs text-gray-500">
                    {item.timestamp ? new Date(item.timestamp).toLocaleDateString() : 'Unknown date'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </main>
  );
}
