var interval = null;

$(document).ready(function () {
    let namespace = "/test";
    let video = document.querySelector("#videoElement");
    let canvas = document.querySelector("#canvasElement");
    let ctx = canvas.getContext('2d');
    let photo = document.getElementById('photo');
    let drawableCanvas = document.querySelector('#drawableCanvas');

    let localMediaStream = null;

    let INIT_FPS = 3;

    let aspectRatio = 0;

    let socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);


    socket.on('output-event', function (data) {
        // img.src = dataURL//data.image_data
        // outputDelayCounter = 0;
        // console.log('response image')
        // if (startTime = 0) {
        //     startTime = performance.now();
        // } else {
        //     let endTime = performance.now()
        //     processTime = endTime - startTime;
        //     startTime = endTime;
        //     console.log(processTime)
        // }
        let serverFps = data.server_fps;
        photo.setAttribute('src', data.image_data);
        if (interval) {
            clearInterval(interval);
            interval = setInterval(() => {
                sendSnapshot();
            }, (1000/serverFps));
            // console.log('create new interval', interval);
        }
        // if (interval) sendSnapshot();
    });

    function sendSnapshot() {
        // console.log('FPS', 1000 / processTime);
        if (!localMediaStream) {
            return;
        }

        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 512, 512);

        let dataURL = canvas.toDataURL('image/jpeg');

        socket.emit('input', dataURL, mask, aspectRatio);

        // socket.emit('output image')

        // var img = new Image();
    }

    socket.on('connect', function () {
        console.log('Connected!');
    });

    // navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    //   video.srcObject = stream;
    //   localMediaStream = stream;
    //
    //   setInterval(function () {
    //     sendSnapshot();
    //   }, 50);
    // }).catch(function(error) {
    //   console.log(error);
    // });

    let URL = window.URL || window.webkitURL
    // let displayMessage = function (message, isError) {
    //     let element = document.querySelector('#message')
    //     element.innerHTML = message
    //     element.className = isError ? 'error' : 'info'
    // }

    let playSelectedFile = function (event) {
        let file = this.files[0]
        console.log(file);
        let type = file.type
        let videoNode = document.querySelector('#videoElement')
        let canPlay = videoNode.canPlayType(type)
        if (canPlay === '') canPlay = 'no'
        let message = 'Can play type "' + type + '": ' + canPlay
        let isError = canPlay === 'no'
        // displayMessage(message, isError)

        if (isError) {
            return
        }

        let fileURL = URL.createObjectURL(file)
        videoNode.src = fileURL
        //   video.srcObject = stream;
        //   localMediaStream = stream;

        mask = [];
        elements = [];

        videoNode.onseeking = function() {
            if (interval) {
                clearInterval(interval);
                interval = setInterval(() => {
                    sendSnapshot();
                }, 1000/INIT_FPS);
                // console.log('create new interval', interval);
            }

        };

        videoNode.onpause = function() {
            // console.log('interval', interval);
            if (interval) {
                clearInterval(interval);
                interval = null;
            }
        };

        videoNode.onended = function() {
            if (interval) {
                clearInterval(interval);
                interval = null;
            }
        };

        videoNode.onplay = function () {
            // Set the source of one <video> element to be a stream from another.
            console.log(videoNode)
            let stream = videoNode.captureStream();
            localMediaStream = stream;
            processTime = 0;

            aspectRatio = (video.videoWidth / video.videoHeight)

            let width = video.offsetWidth;
            let height = video.offsetHeight;
            if (aspectRatio <= (width/height)) {
                width = parseInt(aspectRatio * height);
            } else {
                height = parseInt(width / aspectRatio);
            }
            drawableCanvas.setAttribute('width', width)
            drawableCanvas.setAttribute('height', height)
            setupCanvas('drawableCanvas')

            // send first snapshot to start interval if it's not interval
            if (!interval) {
                interval = setInterval(() => {
                    sendSnapshot();
                }, 1000/INIT_FPS);
                // console.log('create new interval', interval);
            }

            // sendSnapshot();

            // if (!interval) {
            //     interval = true;
            //     sendSnapshot();
            // }

        };
    }


    let inputNode = document.querySelector('#video-input');
    inputNode.addEventListener('change', playSelectedFile, false);

    let showMask = $('#toggle-mask');
    showMask.change(function () {
        if(this.checked) {
            $('#drawableCanvas').show();
        } else {
            $('#drawableCanvas').hide();
        }
    })

});

