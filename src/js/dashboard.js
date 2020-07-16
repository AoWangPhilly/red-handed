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
                pointBackgroundColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgb(255, 99, 132, 0.2)',
                borderColor: 'rgb(255, 99, 132)',
                data: Object.values(crime)
            }]
        },

        // Configuration options go here
        options: {}
    });
});

d3.json("src/data/dashboard/monthlyCrime.json").then(crime => {
    let ctx = document.getElementById('bargraph').getContext('2d');
    let chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(crime),
            datasets: [{
                label: "Crime Count Each Month (2019)",
                backgroundColor: 'rgb(255, 99, 132, 0.5)',
                data: Object.values(crime)
            }]
        },
        options: {}
    });
});

d3.json("src/data/dashboard/typeOfCrimes2019.json").then(crime => {
    let ctx = document.getElementById('pie').getContext('2d');
    let chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(crime),
            datasets: [{
                label: "Type of Crime Count(2019)",
                backgroundColor: poolColors(Object.values(crime)),
                data: Object.values(crime)
            }]
        },
        options: {}
    });
});

const dynamicColors = () => {
    let r = Math.floor(Math.random() * 255) + 150;
    let g = Math.floor(Math.random() * 255) - 100;
    let b = Math.floor(Math.random() * 255) - 100;
    return "rgba(" + r + "," + g + "," + b + ", 0.5)";
}

const poolColors = a => {
    let pool = [];
    for (let i = 0; i < a.length; i++) {
        pool.push(dynamicColors());
    }
    return pool;
}
