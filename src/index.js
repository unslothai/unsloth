import express from 'express';
import modelsRouter from './api/models';

const app = express();
app.use('/api/models', modelsRouter);

app.listen(3000, () => {
  console.log('Server started on port 3000');
});