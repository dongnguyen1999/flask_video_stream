// Get a handle to the player
var player = document.getElementById('videoElement');
var btnPlayPause = document.getElementById('btnPlayPause');
var btnMute = document.getElementById('btnMute');
var progressBar = document.getElementById('progress-bar');
var volumeBar = document.getElementById('volume-bar');
var timelabel = document.getElementById('timelabel');

// Update the video volume
volumeBar.addEventListener("change", function (evt) {
    player.volume = evt.target.value;
});

document.getElementById('btnFullScreen').disabled = true;

// Add a listener for the timeupdate event so we can update the progress bar
player.addEventListener('timeupdate', updateProgressBar, false);

// Add a listener for the play and pause events so the buttons state can be updated
player.addEventListener('play', function () {
    // Change the button to be a pause button
    changeButtonType(btnPlayPause, 'pause');
}, false);

player.addEventListener('pause', function () {
    // Change the button to be a play button
    changeButtonType(btnPlayPause, 'play');
}, false);

player.addEventListener('volumechange', function (e) {
    // Update the button to be mute/unmute
    if (player.muted) changeButtonType(btnMute, 'unmute');
    else changeButtonType(btnMute, 'mute');
}, false);

player.addEventListener('ended', function () {
    this.pause();
}, false);

progressBar.addEventListener("click", seek);

function seek(e) {
    var percent = e.offsetX / this.offsetWidth;
    player.currentTime = percent * player.duration;
    e.target.value = Math.floor(percent / 100);
    e.target.innerHTML = progressBar.value + '% played';
}

function playPauseVideo() {
    if (player.paused || player.ended) {
        // Change the button to a pause button
        changeButtonType(btnPlayPause, 'pause');
        player.play();
    } else {
        // Change the button to a play button
        changeButtonType(btnPlayPause, 'play');
        player.pause();
    }
}

// Stop the current media from playing, and return it to the start position
function stopVideo() {
    player.pause();
    if (player.currentTime) player.currentTime = 0;
}

// Toggles the media player's mute and unmute status
function muteVolume() {
    if (player.muted) {
        // Change the button to a mute button
        changeButtonType(btnMute, 'mute');
        player.muted = false;
    } else {
        // Change the button to an unmute button
        changeButtonType(btnMute, 'unmute');
        player.muted = true;
    }
}

// Replays the media currently loaded in the player
function replayVideo() {
    resetPlayer();
    player.play();
}


const SECONDS_PER_DAY = 86400;
const HOURS_PER_DAY = 24;
/**
 * Convert seconds to HH:MM:SS
 * If seconds exceeds 24 hours, hours will be greater than 24 (30:05:10)
 *
 * @param {number} seconds
 * @returns {string}
 */
const secondsToHms = (seconds) => {
  const days = Math.floor(seconds / SECONDS_PER_DAY);
  const remainderSeconds = seconds % SECONDS_PER_DAY;
  const hms = new Date(remainderSeconds * 1000).toISOString().substring(11, 19);
  return hms.replace(/^(\d+)/, h => `${Number(h) + days * HOURS_PER_DAY}`.padStart(2, '0'));
};

// Update the progress bar
function updateProgressBar() {
    // Work out how much of the media has played via the duration and currentTime parameters
    var percentage = !isNaN(player.duration) ?Math.floor((100 / player.duration) * player.currentTime): 0;

    // Update the progress bar's value
    progressBar.value = percentage;
    timelabel.innerHTML = secondsToHms(parseInt(player.currentTime));
    // Update the progress bar's text (for browsers that don't support the progress element)
    progressBar.innerHTML = percentage + '% played';
}

// Updates a button's title, innerHTML and CSS class
function changeButtonType(btn, value) {
    btn.title = value;
    if (value == 'play') {
        value = '<i class="fas fa-play"></i>';
    }
    if (value == 'pause') {
        value = '<i class="fas fa-pause"></i>';
    }
    btn.innerHTML = value;
    btn.className = value;
}

function resetPlayer() {
    progressBar.value = 0;
    // Move the media back to the start
    player.currentTime = 0;
    // Set the play/pause button to 'play'
    changeButtonType(btnPlayPause, 'play');
}

function exitFullScreen() {
    if (document.exitFullscreen) {
        document.exitFullscreen();
    } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
    } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen();
    } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
    }
}

function toggleFullScreen() {
    //var player = document.getElementById("player");

    if (player.requestFullscreen)
        if (document.fullScreenElement) {
            document.cancelFullScreen();
        } else {
            player.requestFullscreen();
        }
    else if (player.msRequestFullscreen)
        if (document.msFullscreenElement) {
            document.msExitFullscreen();
        } else {
            player.msRequestFullscreen();
        }
    else if (player.mozRequestFullScreen)
        if (document.mozFullScreenElement) {
            document.mozCancelFullScreen();
        } else {
            player.mozRequestFullScreen();
        }
    else if (player.webkitRequestFullscreen)
        if (document.webkitFullscreenElement) {
            document.webkitCancelFullScreen();
        } else {
            player.webkitRequestFullscreen();
        }
    else {
        alert("Fullscreen API is not supported");

    }
}
