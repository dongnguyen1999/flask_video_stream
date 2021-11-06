
function setupSlider(sliderId) {
    $( `#${sliderId}` ).slider({
        create: function() {
            let handle = $(`#${sliderId} .custom-handle`);
            handle.text( $( this ).slider( "value" ));
        },
        slide: function( event, ui ) {
            let handle = $(`#${sliderId} .custom-handle`);
            handle.text( ui.value );
        },
        min: $(`#${sliderId}`).data('min'),
        max: $(`#${sliderId}`).data('max'),
        step: $(`#${sliderId}`).data('step'),
        value: $(`#${sliderId}`).data('init'),
    });
}

function setupRange(rangeId) {
     $( `#${rangeId}` ).slider({
        create: function() {
            let handleLeft = $(`#${rangeId} .custom-handle.left`);
            let handleRight = $(`#${rangeId} .custom-handle.right`);
            let [left, right] = $( this ).slider( "values" );
            handleLeft.text(left);
            handleRight.text(right);
        },
        slide: function( event, ui ) {
            let [left, right] = ui.values;
            let handleLeft = $(`#${rangeId} .custom-handle.left`);
            let handleRight = $(`#${rangeId} .custom-handle.right`);
            handleLeft.text(left);
            handleRight.text(right);
        },
        range: true,
        min: $(`#${rangeId}`).data('min'),
        max: $(`#${rangeId}`).data('max'),
        step: $(`#${rangeId}`).data('step'),
        values: $(`#${rangeId}`).data('init'),
    });
}

$(function () {
    $('#loading-model').hide();

    $('input[type=radio][name=crowd_model]').change(function() {
        let value = $(this).val();
        let currentCountModel = $('input[type=radio][name=count_model]:checked').val();
        $('#loading-model').show();
        $.ajax({
            type: "POST",
            url: "/change_model",
            data: { crowd_model: value, count_model: currentCountModel },
            success: function (response) {
                console.log(response)
                if (!response.error) {
                    location.reload();
                }
                $('#loading-model').hide();
            },
            error: function(error) {
                console.log(error)
            }
        });
    });

    $('input[type=radio][name=count_model]').change(function() {
        let value = $(this).val();
        let currentCrowdModel = $('input[type=radio][name=crowd_model]:checked').val();

        $('#loading-model').show();
        $.ajax({
            type: "POST",
            url: "/change_model",
            data: { crowd_model: currentCrowdModel, count_model: value },
            success: function (response) {
                console.log(response)
                if (!response.error) {
                    location.reload();
                }
                $('#loading-model').hide();
            },
            error: function(error) {
                console.log(error)
            }
        });
    });

    setupSlider('slider-classification');
    setupSlider('slider-count');
    setupRange('slider-range-crowd');
    setupRange('slider-range-count');

    // $( "#amount" ).val( "$" + $( "#slider-range" ).slider( "values", 0 ) +
    //     " - $" + $( "#slider-range" ).slider( "values", 1 ) );
    // } );

})