var interval = null;
var change_video = false;

$(document).ready(function () {
    let namespace = "/test";
    let video = document.querySelector("#videoElement");
    let canvas = document.querySelector("#canvasElement");
    let ctx = canvas.getContext('2d');
    let photo = document.getElementById('photo');
    let drawableCanvas = document.querySelector('#drawableCanvas');

    let localMediaStream = null;

    let INIT_FPS = 1;

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
        let result = JSON.parse(data.result);

        let serverFps = parseFloat(result.fps);
        let clsFps = parseFloat(result.cls_fps).toFixed(1);
        let dtFps = result.dt_fps !== undefined? parseFloat(result.dt_fps).toFixed(1): undefined;
        let bsFps = result.bs_fps !== undefined? parseFloat(result.bs_fps).toFixed(1): undefined;

        let totalCount = parseInt(result.total_count);

        let countScore = result.count_score !== undefined? parseFloat(result.count_score).toFixed(2): undefined;
        let diffRate = result.diff_rate !== undefined? parseFloat(result.diff_rate).toFixed(2): undefined;
        let label = parseInt(result.label);
        let labelName = result.label_name;

        photo.setAttribute('src', data.image_data);

        $('#fps-info span').text(serverFps.toFixed(1));
        $('#cls-fps-info span').text(clsFps);
        if (dtFps) {
            $('#bs-fps-info').hide();
            $('#count-fps-info span').text(dtFps);
            $('#count-fps-info').show();
        }

        if (bsFps) {
            $('#count-fps-info').hide();
            $('#bs-fps-info span').text(bsFps);
            $('#bs-fps-info').show();
        }

        if (dtFps) {
            $('#counting-result .total span').text(totalCount);
            let count = JSON.parse(result.count);
            let countColors = JSON.parse(result.count_label_colors.replaceAll("'", '"'));
            const labelMap = ['2-wheel', '4-wheel', 'priority']
            for (let i = 0; i < 3; i++) {
                let counti = parseInt(count[i]);
                let color = countColors[i];
                console.log(color);
                $(`#counting-result .${labelMap[i]} span`).text(counti)
                $(`#counting-result .${labelMap[i]}`).css('color', color);
            }
            $('#counting-result').show();
        } else $('#counting-result').hide();

        if (countScore) {
            $('#predict-result .moving-rate').hide();
            $('#predict-result .count-score span').text(countScore);
            $('#predict-result .count-score').show();
        }

        if (diffRate) {
            $('#predict-result .count-score').hide();
            $('#predict-result .moving-rate span').text(diffRate);
            $('#predict-result .moving-rate').show();
        }

        console.log(label);
        let status = label < 3? 'Not Crowded': 'Crowded';
        let labelColor = label < 3? '#009b00': '#cb0000';
        $('#predict-result .status-label span').text(status).css('color', labelColor);
        $('#predict-result .label span').text(labelName).css('color', labelColor);

        if (interval) {
            clearInterval(interval);
            if (!blockInterval) {
                interval = setInterval(() => {
                    sendSnapshot();
                }, (1000/serverFps + 200));
            }

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

           // 'count_conf_threshold': model.count_conf_threshold,
           // 'classify_conf_threshold': model.classify_conf_threshold,
           // 'count_thresholds': model.count_thresholds,
           // 'crowd_thresholds': model.crowd_thresholds,

        let count_conf_threshold = $('#slider-count').slider('value');
        let classify_conf_threshold = $('#slider-classification').slider('value');
        let count_thresholds = $('#slider-range-count').slider('values');
        let crowd_thresholds = $('#slider-range-crowd').slider('values');

        let show_bounding_box = $('#toggle-show-box').is(":checked");
        let show_diff_mask = $('#toggle-show-diff').is(":checked");
        let show_foreground_mask = $('#toggle-show-foreground').is(":checked");

        if (show_diff_mask || show_foreground_mask) {
            $('#toggle-mask').prop('checked', false); // Unchecks it
            $('#toggle-mask').prop('disabled', true); // Unchecks it
            $('#drawableCanvas').hide();
        } else {
            $('#toggle-mask').prop('disabled', false); // Unchecks it
        }


        let config = {
            count_conf_threshold: count_conf_threshold,
            classify_conf_threshold: classify_conf_threshold,
            count_thresholds: count_thresholds,
            crowd_thresholds: crowd_thresholds,
            show_bounding_box: show_bounding_box,
            show_diff_mask: show_diff_mask,
            show_foreground_mask: show_foreground_mask,
            aspect_ratio: aspectRatio,
            reset_session: change_video,
        }

        console.log(change_video)
        socket.emit('input', dataURL, mask, JSON.stringify(config));
        if (change_video) change_video = false;

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

        change_video = true;

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
            change_video = true;
            if (interval) {
                clearInterval(interval);
                if (!blockInterval) {
                    interval = setInterval(() => {
                        sendSnapshot();
                    }, 1000/INIT_FPS);
                }
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
            if (!interval && !blockInterval) {
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

