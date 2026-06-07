import React, { useState } from 'react';

const ModelSelector = () => {
  const [draftModel, setDraftModel] = useState(null);

  const handleDraftModelChange = (event) => {
    setDraftModel(event.target.value);
  };

  return (
    <div>
      <label>
        Select Draft Model:
        <select value={draftModel} onChange={handleDraftModelChange}>
          <option value="">None</option>
          <option value="Qwen3.5 0.8b">Qwen3.5 0.8b</option>
          {/* Add more draft models as options */}
        </select>
      </label>
    </div>
  );
};

export default ModelSelector;