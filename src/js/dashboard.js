d3.json("src/data/dashboard/crimePerYear.json").then(crime => {
    let ctx = document.getElementById('crimeOverYear').getContext('2d');
    let chart = new Chart(ctx, {
        // The type of chart we want to create
        type: 'line',

        // The data for our dataset
        data: {
            labels: Object.keys(crime),
            datasets: [{
                label: 'Philadelphia Crime (2006-2020)',
                backgroundColor: 'rgb(255, 99, 132)',
                borderColor: 'rgb(255, 99, 132)',
                data: Object.values(crime)
            }]
        },

        // Configuration options go here
        options: {}
    });
});