const express = require('express');
const router = express.Router();
const axios = require('axios');
const fs = require('fs');

router.get('/', async (req, res) => {
  const models = await fs.readdirSync('./models');
  res.json(models.map((model) => ({ name: model, sha: fs.readFileSync(`./models/${model}/sha`, 'utf8') })));
});

router.delete('/:modelName', async (req, res) => {
  const modelName = req.params.modelName;
  await fs.rmdirSync(`./models/${modelName}`, { recursive: true });
  res.json({ message: `Model ${modelName} deleted successfully` });
});

router.get('/:modelName', async (req, res) => {
  const modelName = req.params.modelName;
  const response = await axios.get(`https://huggingface.co/api/models/${modelName}`);
  const modelData = response.data;
  await fs.mkdirSync(`./models/${modelName}`, { recursive: true });
  await fs.writeFileSync(`./models/${modelName}/sha`, modelData.sha);
  res.json(modelData);
});

module.exports = router;