// defines a TF model load function
const loadModel = async function loadModel() {
    // loads the model
    model = await tf.loadLayersModel('mnist_tf_keras_js_model/model.json');
    // warm start the model. speeds up the first inference
    model.predict(tf.zeros([1, 28, 28]));
    return model
}
const MODEL = loadModel();

const cropImage = function cropImage(image) {
    const BLACK = 0;
    const WIDTH = 28;
    const HEIGHT = 28;
    let xMin = WIDTH;
    let yMin = HEIGHT;
    let xMax = 0;
    let yMax = 0;
    for (let row = 0; row < HEIGHT; ++row) {
        for (let col = 0; col < WIDTH; ++col) {
            if (image[col + row * WIDTH] != BLACK) {
                xMin = (col < xMin ? col : xMin);
                xMax = (col > xMax ? col : xMax);

                yMin = (row < yMin ? row : yMin);
                yMax = (row > yMax ? row : yMax);
            }
        }
    }
    yMax = (yMax < HEIGHT - 1 ? yMax + 1 : yMax);
    xMax = (xMax < WIDTH - 1 ? xMax + 1 : xMax);
    imgOutWidth = xMax - xMin;
    imgOutHeight = yMax - yMin;
    const imgOut = new Array(imgOutWidth * imgOutHeight);
    let ind = 0;
    for (let row = yMin; row < yMax; ++row) {
        for (let col = xMin; col < xMax; ++col) {
            imgOut[ind] = image[col + row * WIDTH];
            ++ind;
        }
    }
    return { image: imgOut, width: imgOutWidth, height: imgOutHeight };
}


const strechImage = function strechImage(imgArray, width, height) {
    const W_OUT = 28;
    const H_OUT = 28;
    const tmp = new Array(W_OUT * H_OUT);
    const xScale = W_OUT / width;
    const yScale = H_OUT / height;

    // horizontal rescale
    for (let row = 0; row < height; ++row) {
        for (let pixel = 0; pixel < width; ++pixel) {
            let thisPixel = Math.ceil(xScale * pixel);
            let nextPixel = Math.ceil(xScale * (pixel + 1));

            for (let i = thisPixel; i < nextPixel; ++i) {
                if (i + row * W_OUT >= W_OUT * H_OUT) {
                    break;
                }
                tmp[i + row * W_OUT] = imgArray[pixel + row * width];
            }
        }
    }

    const out = new Array(W_OUT * H_OUT);
    // vertical rescale
    for (let col = 0; col < W_OUT; ++col) {
        for (let pixel = 0; pixel < height; ++pixel) {
            let thisPixel = Math.ceil(yScale * pixel);
            let nextPixel = Math.ceil(yScale * (pixel + 1));

            for (let i = thisPixel; i < nextPixel; ++i) {
                if (col + i * W_OUT >= W_OUT * H_OUT) {
                    break;
                }
                out[col + i * W_OUT] = tmp[col + pixel * W_OUT];
            }
        }
    }
    return out;
}

const guessTheNumberTf = async function guessTheNumberTf() {
    const imgWidth = 280;
    const imgHeight = 280;
    let imgData = ctx.getImageData(0, 0, imgWidth, imgHeight);
    const R = getSingleRGBAChannel(imgData.data, 0);
    const rescaledImage = rescaleTo28x28(R);
    const croppedImage = cropImage(rescaledImage);
    const strechedImage = strechImage(croppedImage.image, croppedImage.width, croppedImage.height);
    const normalizedImage = strechedImage.map(el => el / 255);
    const tensor = tf.tensor(normalizedImage, [1, 28, 28]);
    const predictionArray = model.predict(tensor).dataSync();

    let nr = 0;
    let max = predictionArray[0];
    let min = predictionArray[0];
    for (let i = 1; i < 10; ++i) {
        if (predictionArray[i] > max) {
            nr = i;
            max = predictionArray[i];
        }
        if (predictionArray[i] < min) {
            min = predictionArray[i];
        }
    }
    let startIndex = (nr > 0 ? 0 : 1);
    let nr2 = startIndex;
    let secondLargest = predictionArray[startIndex];
    for (let i = 0; i < 10; ++i) {
        if (i == nr) continue;
        if (predictionArray[i] > secondLargest) {
            nr2 = i;
            secondLargest = predictionArray[i];
        }
    }

    const txt = document.getElementById("bot-field-2");
    if (max - secondLargest > 3) {
        txt.innerHTML = "The number is: " + nr;
    } else {
        txt.innerHTML = "Doesn't look like a number."
    }

    let element = "";
    for (let i = 0; i < predictionArray.length; ++i) {
        const normalizedValue = (predictionArray[i] - max) / (max - min) + 1;
        element += "<div class='prob'>" + Math.round(normalizedValue * 100) / 100 + "</div>";
    }
    document.getElementById("probabilities-2").innerHTML = element
}

document.getElementById('guess-btn').addEventListener('click', guessTheNumberTf);