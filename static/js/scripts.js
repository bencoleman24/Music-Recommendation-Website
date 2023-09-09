$(document).ready(function() {
    // Detect change in recommendation model dropdown
    $('#modelTypeSelector').change(function() {
        // Hide all input sections initially
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

    // Submit the form
    $('form').submit(function(e) {
        e.preventDefault();

        $('#loadingMessage').show();

        $.post("/get_recommendations", $(this).serialize(), function(data) {
            $('#loadingMessage').hide();
            
            // Display the recommendations
            $('#recommendationsList').empty();
            if (data.status === "success" && data.data && data.data.length && data.links && data.links.length === data.data.length) {
                for(let i = 0; i < data.data.length; i++) {
                    $('#recommendationsList').append('<li class="list-group-item">' + data.data[i] + ' - <a href="' + data.links[i] + '" target="_blank">Link</a></li>');
                }
                $('#resultsSection').show();
                $('#feedbackSection').show();

                // Hide the "Generate Recommendations" button
                $('#generateBtn').hide();
            } else {
                alert(data.message);  // or another user-friendly way to show the error message
            }
        });
    });

    // Restart button functionality
    $('#restartButton').click(function() {
        $('#resultsSection').hide();
        $('#feedbackSection').hide();
        $('#initialPrompt').show();

        // Show the "Generate Recommendations" button
        $('#generateBtn').show();
    });

    // Handle feedback submission if needed
    $('#submitFeedback').click(function() {
        $('#feedbackSubmittedMessage').show();
    
        // Send feedback to server TODO
    });
    
});
