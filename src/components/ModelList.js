import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ModelList() {
  const [models, setModels] = useState([]);
  const [updatesAvailable, setUpdatesAvailable] = useState({});

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    const response = await axios.get('/api/models');
    setModels(response.data);
    checkForUpdates(response.data);
  };

  const checkForUpdates = async (models) => {
    const updates = {};
    for (const model of models) {
      const response = await axios.get(`https://huggingface.co/api/models/${model.name}`);
      if (response.data.sha === model.sha) {
        updates[model.name] = false;
      } else {
        updates[model.name] = true;
      }
    }
    setUpdatesAvailable(updates);
  };

  const handleDelete = async (modelName) => {
    if (window.confirm(`Are you sure you want to delete ${modelName}?`)) {
      await axios.delete(`/api/models/${modelName}`);
      fetchModels();
    }
  };

  const handleUpdate = async (modelName) => {
    await axios.get(`https://huggingface.co/api/models/${modelName}`);
    fetchModels();
  };

  return (
    <div>
      <h2>Models</h2>
      <ul>
        {models.map((model) => (
          <li key={model.name}>
            {model.name}
            {updatesAvailable[model.name] && <span> (Update available)</span>}
            <button onClick={() => handleDelete(model.name)}>Delete</button>
            <button onClick={() => handleUpdate(model.name)}>Update</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ModelList;