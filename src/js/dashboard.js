"use strict";

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
                pointBackgroundColor: 'rgb(255, 0, 0)',
                backgroundColor: 'rgb(255, 0, 0, 0.5)',
                borderColor: 'rgb(255, 50, 50)',
                data: Object.values(crime),
            }]
        },

        // Configuration options go here
        options: {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Number of Crimes"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Years"
                    }
                }]
            }
        }
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
                backgroundColor: 'rgb(255, 0, 0, 0.70)',
                data: Object.values(crime)
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Number of Crimes"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Months"
                    }
                }]
            }
        }
    });
});

d3.json("src/data/dashboard/typeOfCrimes2019.json").then(crime => {
    let ctx = document.getElementById('pie').getContext('2d');
    let keys = Object.keys(crime);
    keys.sort((a, b) => crime[b] - crime[a]);

    let values = [];
    let keyLength = keys.length;

    for (let i = 0; i < keyLength; i++) {
        values.push(crime[keys[i]]);
    }

    let chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: keys,
            datasets: [{
                label: "Type of Crime Count(2019)",
                backgroundColor: poolColors(Object.values(crime)),
                data: values
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Number of Crimes"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Types of Crimes"
                    }
                }]
            }
        }
    });
});

const dynamicColors = () => {
    let r = 255;
    let g = Math.floor(Math.random() * 100);
    let b = Math.floor(Math.random() * 100);
    return `rgba(${r}, ${g}, ${b}, 0.8)`;
}

const poolColors = arr => {
    let pool = [];
    for (let i = 0; i < arr.length; i++) {
        pool.push(dynamicColors());
    }
    return pool;
}

d3.json("src/data/dashboard/zipcodeCrime.json").then(crime => {
    let ctx = document.getElementById('zipcode').getContext('2d');
    let chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(crime),
            datasets: [{
                label: "Crime Count in Each Zipcode(2019)",
                backgroundColor: 'rgb(255, 0, 0, 0.70)',
                data: Object.values(crime)
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Number of Crimes"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Zipcodes"
                    }
                }]
            }
        }
    });
});

d3.json("src/data/dashboard/medianIncome.json").then(crime => {
    console.log(crime);
    let data = [];
    for (let income in crime) {
        data.push({
            x: income,
            y: crime[income]
        })
    }
    let ctx = document.getElementById('income').getContext('2d');
    let scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Scatter Dataset',
                backgroundColor: 'rgb(255, 0, 0, 0.70)',
                data: data
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Number of Crimes"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Median Household-Income per Zipcode"
                    }
                }]
            }
        }
    });
});

d3.json("src/data/dashboard/populationDensity.json").then(crime => {
    let data = [];
    for (let pop in crime) {
        data.push({
            x: pop,
            y: crime[pop]
        })
    }
    let ctx = document.getElementById('density').getContext('2d');
    let scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Scatter Dataset',
                backgroundColor: 'rgb(255, 0, 0, 0.70)',
                data: data
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Number of Crimes"
                    }
                }],
                xAxes: [{
                    scaleLabel: {
                        display: true,
                        labelString: "Population Density per Zipcode"
                    }
                }]
            }
        }
    });
});