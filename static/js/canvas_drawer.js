var mask = [];
var elements = [];

var mouseDownEvent = null;
var mouseUpEvent = null;
var mouseMoveEvent = null;
var keyUpEvent = null;

function setupCanvas(canvasId) {
    $('#drawableCanvas').show();
    let elem = document.getElementById(canvasId);
    let root = document.getElementById('player');
    let col = document.querySelector('.col-md-8');
    console.log(col.offsetLeft, col.clientLeft)
    console.log(col.offsetTop, col.clientTop)
    let elemLeft = col.offsetLeft + root.offsetLeft + 16 + elem.offsetLeft + elem.clientLeft;
    let elemTop = col.offsetTop + root.offsetTop+ 16 + elem.offsetTop + elem.clientTop;
    let context = elem.getContext('2d');
    let pointRadius = 8;
    let pointColor = '#3542fc';
    let lineColor = '#fc3535';
    let lineWidth = 4;
    let idCounter = 4;
    let draggingElement = null;

    let canvasWidth = elem.getAttribute('width');
    let canvasHeight = elem.getAttribute('height');

    elements = elements.length? elements : [
        {id: 0, x: 1, y:1},
        {id: 1, x: canvasWidth-1, y:1},
        {id: 2, x: canvasWidth-1, y:canvasHeight-1},
        {id: 3, x: 1, y:canvasHeight-1},
    ];

    function check_inside_circle(x, y, ix, iy, r) {
        let dist_points = (x - ix) * (x - ix) + (y - iy) * (y - iy);
        r *= r;
        if (dist_points < r) {
            return true;
        }
        return false;
    }

    function findElement(x, y) {
        for (let element of elements) {
            if (check_inside_circle(x, y, element.x, element.y, pointRadius)) {
                return element;
            }
        }
        return null;
    }

    function findTopLeft(elements){
        if (!elements.length) return null;
        let minY = Infinity;
        for (let point of elements) {
          if (point.y < minY) {
            minY = point.y;
          }
        }
        let minYElements = elements.filter(e => e.y == minY);
        let topleft;
        let minX = Infinity;
        for (let point of minYElements) {
          if (point.x < minX) {
            minX = point.x;
            topleft = point;
          }
        }
        return topleft;
    }

    function sortElements(elements) {
        if (elements.length < 3) return elements;
        let points = [...elements];
        points.sort((a,b) => a.y - b.y);
        const cy = (points[0].y + points[points.length-1].y) / 2;

        points.sort((a,b)=>b.x - a.x);
        const cx = (points[0].x + points[points.length -1].x) / 2;

        const center = {x:cx,y:cy};

        let startAng;
        points.forEach(point => {
            let ang = Math.atan2(point.y - center.y,point.x - center.x);
            if(!startAng){ startAng = ang }
            else {
                 if(ang < startAng){  // ensure that all points are clockwise of the start point
                     ang += Math.PI * 2;
                 }
            }
            point.angle = ang; // add the angle to the point
        });

        points.sort((a,b)=> a.angle - b.angle);
        return points;
    }

    function renderElements(elements) {
        elements = sortElements(elements);
        if (elements.length < 3) mask = [[1,1], [511, 1], [511, 511], [1, 511]];
        else mask = elements.map(element => ([Math.floor((element.x * 512) / canvasWidth), Math.floor((element.y * 512) / canvasHeight)]))
        context.clearRect(0, 0, elem.width, elem.height);
        console.log('elements', elements)
        if (elements.length > 2) {
            context.strokeStyle = lineColor;
            context.lineWidth = lineWidth;

            context.beginPath();
            context.moveTo(elements[0].x, elements[0].y);

            for (let i = 1; i < elements.length; i++){
                context.lineTo(elements[i].x, elements[i].y);
            }

            context.lineTo(elements[0].x, elements[0].y);
            context.closePath();
            context.stroke();
        }
        for (let element of elements) {
            context.beginPath();
            context.arc(element.x, element.y, pointRadius, 0, 2 * Math.PI);
            context.fillStyle = pointColor;
            context.fill();
            context.closePath();
        }

    }

    function mouseDownListener(event) {
        event.stopPropagation();
        let x = event.pageX - elemLeft;
        let y = event.pageY - elemTop;
        // console.log(x, y);
        let selected = findElement(x, y);
        if (selected) {
            draggingElement = selected;
        } else {
            elements.push({
                id: idCounter++,
                x: x,
                y: y,
            });
            renderElements(elements);
        }
    }

    if (mouseDownEvent) elem.removeEventListener('mousedown', mouseDownEvent)
    // Add event listener for `click` events.
    elem.addEventListener('mousedown', mouseDownListener);
    mouseDownEvent = mouseDownListener;

    function mouseUpListener(event) {
        event.stopPropagation();
        let x = event.pageX - elemLeft;
        let y = event.pageY - elemTop;
        // console.log(x, y);
        if (draggingElement) {
            draggingElement = null;
        }
    }

    if (mouseUpEvent) elem.removeEventListener('mouseup', mouseUpEvent);
    elem.addEventListener('mouseup', mouseUpListener);
    mouseUpEvent = mouseUpListener;

    function mouseMoveListener(event) {
        let x = event.pageX - elemLeft;
        let y = event.pageY - elemTop;
        // console.log(x, y);
        if (draggingElement) {
            draggingElement.x = x;
            draggingElement.y = y;
            renderElements(elements);
        }
    }

    if (mouseMoveEvent) elem.removeEventListener('mousemove', mouseMoveEvent);
    elem.addEventListener('mousemove', mouseMoveListener);
    mouseMoveEvent = mouseMoveListener;

    function keyUpListener(event) {
        if (event.key === "Delete" && draggingElement) {
            elements = elements.filter(item => item.id != draggingElement.id)
            draggingElement = null;
            renderElements(elements);
        }
    }

    if (keyUpEvent) elem.removeEventListener('keyup', keyUpEvent);
    document.addEventListener('keyup', keyUpListener);
    keyUpEvent = keyUpListener;

    renderElements(elements);
    $('#drawableCanvas').hide();
    $('#toggle-mask').prop( "checked", false);
}
