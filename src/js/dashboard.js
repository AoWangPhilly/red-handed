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
                lineTension: 0.3,
                pointRadius: 5,
                pointHoverRadius: 5,
                pointBorderWidth: 2,
                pointHitRadius: 20,
                pointBorderColor: "rgba(255,255,255,0.8)",
                pointBackgroundColor:'rgb(255, 99, 132)',
                backgroundColor: 'rgb(255, 99, 132, 0.2)',
                borderColor: 'rgb(255, 99, 132)',
                data: Object.values(crime)
            }]
        },

        // Configuration options go here
        options: {}
    });
});