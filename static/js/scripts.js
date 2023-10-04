$(document).ready(function() {
    $('#modelTypeSelector').change(function() {
        $('#inputSection').hide();
        $('#popularityInputs').hide();
        $('#MLInputs').hide();

        let selectedModel = $(this).val();

        switch(selectedModel) {
            case 'popularity_based':
                $('#inputSection').show();
                $('#popularityInputs').show();
                break;
            case 'cosine_similarity':
            case 'kmeans':
            case 'nn':
                $('#inputSection').show();
                $('#MLInputs').show();
                break;
        }
    });

    $('form').submit(function(e) {
        e.preventDefault();

        $('#loadingMessage').show();

        $.post("/get_recommendations", $(this).serialize(), function(data) {
            $('#loadingMessage').hide();
            
            if (data.status === "failure" && data.message.includes("Please wait")) {
                alert(data.message);
                return;
            }
            
            $('#recommendationsList').empty();
            if (data.status === "success" && data.data && data.data.length && data.links && data.links.length === data.data.length) {
                for(let i = 0; i < data.data.length; i++) {
                    $('#recommendationsList').append('<li class="list-group-item">' + data.data[i] + ' - <a href="' + data.links[i] + '" target="_blank">Song Link</a></li>');
                }
                $('#resultsSection').show();
                
                $('#recommendationTitle').show();
                
                $('#feedbackSection').show();

                $('#generateBtn').hide();
                
                // Re-enable the "Submit Feedback" button when new recommendations are displayed
                $('#submitFeedback').prop('disabled', false);

            } else {
                alert(data.message);  
            }
        });
    });

    $('#restartButton').click(function() {
        $('#resultsSection').hide();
        $('#recommendationTitle').hide();
        $('#feedbackSection').hide();
        $('#initialPrompt').show();

        $('#generateBtn').show();
    });

    $('#submitFeedback').click(function() {
        let feedback = $('#feedbackRating').val();
        let form_data = $("form").serializeArray();
        let input_data = "";
        form_data.forEach(item => {
            if(item.name === "artist" && item.value !== "") {
                input_data += "Artist: " + item.value + ", ";
            }
            if(item.name === "track_name" && item.value !== "") {
                input_data += "Track: " + item.value;
            }
        });
        let model = $('#modelTypeSelector').val();
        let output = $('#recommendationsList').text();  

        $.post("/submit_feedback", {
            feedback: feedback,
            input_data: input_data,
            model: model,
            output: output
        }, function(data) {
            if (data.status === "success") {
                $('#feedbackSubmittedMessage').show(500);
                setTimeout(function() {
                    $('#feedbackSubmittedMessage').hide();
                }, 5000);
                $('#submitFeedback').prop('disabled', true);  
            } else {
                alert(data.message);
            }
        });
    });

});
