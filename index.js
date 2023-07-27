/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const CLASSES_NAMES = {

  0: 'Alpine sea holly',
  1: 'Anthurium',
  2: 'Artichoke',
  3: 'Azalea',
  4: 'Balloon Flower',
  5: 'Barberton Daisy',
  6: 'Bee Balm',
  7: 'Bird of paradise',
  8: 'Bishop of llandaf',
  9: 'Black-eyed susan',
  10: 'Blackberry lily',
  11: 'Blanket Flower',
  12: 'Bolero deep blue',
  13: 'Bougainvillea',
  14: 'Bromelia',
  15: 'Buttercup',
  16: 'Californian poppy',
  17: 'Camellia',
  18: 'Canna lily',
  19: 'Canterbury bells',
  20: 'Cape flower',
  21: 'Carnation',
  22: 'Cautleya Spicata',
  23: 'Clematis',
  24: 'Colts foot',
  25: 'Columbine',
  26: 'Common dandelian',
  27: 'Common tulip',
  28: 'Corn poppy',
  29: 'Cosmos',
  30: 'Cyclamen',
  31: 'Daffodil',
  32: 'Daisy',
  33: 'Desert-rose',
  34: 'Fire lily',
  35: 'Foxglove',
  36: 'Frangipani',
  37: 'Fritillary',
  38: 'Garden phlox',
  39: 'Gaura',
  40: 'Gazania',
  41: 'Geranium',
  42: 'Giant white arum lily',
  43: 'Globe thistle',
  44: 'Globe-flower',
  45: 'Grape hyacinth',
  46: 'Great masterwort',
  47: 'Hard-leaved pocket orchid',
  48: 'Hibiscus',
  49: 'Hippeastrum',
  50: 'Iris',
  51: 'Japanese anemone',
  52: 'King protea',
  53: 'Lenten rose',
  54: 'Lilac hibiscus',
  55: 'Lotus',
  56: 'Love in the mist',
  57: 'Magnolia',
  58: 'Mallow',
  59: 'Marigold',
  60: 'Mexican petunia',
  61: 'Monkshood',
  62: 'Moon orchid',
  63: 'Morning glory',
  64: 'Orange dahlia',
  65: 'Osteospermum',
  66: 'Passion flower',
  67: 'Peruvian lily',
  68: 'Petunia',
  69: 'Pincushion flower',
  70: 'Pink primrose',
  71: 'Pink quill',
  72: 'Pink-yellow dahlia',
  73: 'Poinsettia',
  74: 'Primula',
  75: 'Prince of wales feathers',
  76: 'Purple coneflower',
  77: 'Red ginger',
  78: 'Rose',
  79: 'Ruby-lipped cattleya',
  80: 'Siam tulip',
  81: 'Silverbush',
  82: 'Snapdragon',
  83: 'Spear thistle',
  84: 'Spring crocus',
  85: 'Stemless gentian',
  86: 'Sunflower',
  87: 'Sweet pea',
  88: 'Sweet william',
  89: 'Sword lily',
  90: 'Thorn apple',
  91: 'Tiger lily',
  92: 'Toad lily',
  93: 'Tree mallow',
  94: 'Tree poppy',
  95: 'Trumpet creeper',
  96: 'Wallflower',
  97: 'Water lily',
  98: 'Watercress',
  99: 'Wild geranium',
  100: 'Wild pansy',
  101: 'Wild rose',
  102: 'Windflower',
  103: 'Yellow iris',
  


 }

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'model_tfjs';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 3;

let mobilenet;
const mobilenetDemo = async () => {
  status('Loading model...');

  // mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH, {fromTFHub: true});
  mobilenet = await tf.loadLayersModel('model_tfjs/model.json');

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = document.getElementById('cat');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.cast(tf.browser.fromPixels(imgElement), 'float32');

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].X
    // const normalized = img.sub(offset).div(offset);
    const normalized = img;

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: CLASSES_NAMES[topkIndices[i]],
      pred: topkIndices[i],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    // const predElement = document.createElement('div');
    // classElement.className = 'cell';
    // classElement.innerText = classes[i].pred;
    // row.appendChild(classElement);
    var description = "";
    for (var j = 0; j < jsonData.length; j++){
      if (jsonData[j]["Flower Type"] === classes[i].className) {
        description = jsonData[j]["Description"];}
    }
    const descriptionElement = document.createElement('div');
    descriptionElement.className = 'cell';
    descriptionElement.innerText = description;
    row.appendChild(descriptionElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();

async function fetchJSONData() {
  try {
    const response = await fetch('data.json'); 
    jsonData = await response.json();
    console.log('JSON data:', jsonData);
    mobilenetDemo(); 
  } catch (error) {
    console.error('Error fetching JSON data:', error);
  }
}
fetchJSONData();
