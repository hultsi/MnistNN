// const canvas = document.getElementById('myCanvas');
// const ctx = canvas.getContext('2d');
let wasmReady = false;
let guess_number = null;

Module = {
    preRun: [],
    postRun: [],
    onRuntimeInitialized: () => {
        guess_number = Module.cwrap('guess_number', null, ["number", "number", "number", "number"]);
        wasmReady = true;
    }
};

const guessTheNumber = function guessTheNumber() {
    if (wasmReady) {
        const imgWidth = 280;
        const imgHeight = 280;
        let imgData = ctx.getImageData(0, 0, imgWidth, imgHeight);
        const R = getSingleRGBAChannel(imgData.data, 0);
        const IN_ARR = rescaleTo28x28(R);

        const SIZE_IN = 784;
        const SIZE_OUT = 10;
        if (IN_ARR.length != SIZE_IN) {
            console.log("INPUT ARRAY SIZE IS ", IN_ARR.length, " BUT REQUIRED SIZE IS ", SIZE_IN);
            return;
        }
        const inArr = new Int32Array(IN_ARR);
        const bytes = inArr.BYTES_PER_ELEMENT;

        const ptr0 = Module._malloc(SIZE_IN * bytes);
        const ptr1 = Module._malloc(SIZE_OUT * bytes);

        Module.HEAP32.set(inArr, ptr0 / bytes);
        guess_number(ptr0, SIZE_IN, ptr1, SIZE_OUT);
        const outArr = new Float32Array(Module.HEAPF32.buffer, ptr1, SIZE_OUT);

        let nr = 0;
        let max = outArr[0];
        let min = outArr[0];
        for (let i = 1; i < 10; ++i) {
            if (outArr[i] > max) {
                nr = i;
                max = outArr[i];
            }
            if (outArr[i] < min) {
                min = outArr[i];
            }
        }
        let startIndex = (nr > 0 ? 0 : 1);
        let nr2 = startIndex;
        let secondLargest = outArr[startIndex];
        for (let i = 0; i < 10; ++i) {
            if (i == nr) continue;
            if (outArr[i] > secondLargest) {
                nr2 = i;
                secondLargest = outArr[i];
            }
        }

        const txt = document.getElementById("bot-field");
        if (max - secondLargest > 0.1 && max > 0.5) {
            txt.innerHTML = "The number is: " + nr;
        } else {
            txt.innerHTML = "Doesn't look like a number."
        }

        let element = "";
        for (let i = 0; i < outArr.length; ++i) {
            const normalizedValue = (outArr[i] - max) / (max - min) + 1;
            element += "<div class='prob'>" + Math.round(normalizedValue * 100) / 100 + "</div>";
        }
        document.getElementById("probabilities").innerHTML = element

        Module._free(ptr0);
        Module._free(ptr1);
    }
}

const getSingleRGBAChannel = function getSingleRGBAChannel(rgba, channel = 0) {
    const out = new Array(rgba.length / 4);
    let rgbaInd = channel;
    for (let i = 0; i < out.length; ++i) {
        out[i] = rgba[rgbaInd];
        rgbaInd += 4;
    }
    return out;
}

const rescaleTo28x28 = function rescaleTo28x28(arr) {
    let out = new Array(784).fill(0);
    let ind1 = 0;
    let ind2 = 0;
    const cols = 280;
    while (ind2 < 784) {
        for (let i = 0; i < 10; ++i) {
            for (let k = 0; k < 10; ++k) {
                out[ind2] += arr[ind1 + cols * k + i];
            }
        }
        out[ind2] /= 100;
        if ((ind1 + 10) % 280 == 0) {
            ind1 += 280 * 9 + 10;
        } else {
            ind1 += 10;
        }
        ++ind2;
    }
    return out;
}

document.getElementById('guess-btn').addEventListener('click', guessTheNumber);