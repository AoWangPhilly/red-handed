"use strict";

let map;

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: {
            lat: 39.9909,
            lng: -75.1115
        },
        zoom: 11,
        styles: [{
                elementType: 'geometry',
                stylers: [{
                    color: '#242f3e'
                }]
            },
            {
                elementType: 'labels.text.stroke',
                stylers: [{
                    color: '#242f3e'
                }]
            },
            {
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#746855'
                }]
            },
            {
                featureType: 'administrative.locality',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#d59563'
                }]
            },
            {
                featureType: 'poi',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#d59563'
                }]
            },
            {
                featureType: 'poi.park',
                elementType: 'geometry',
                stylers: [{
                    color: '#263c3f'
                }]
            },
            {
                featureType: 'poi.park',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#6b9a76'
                }]
            },
            {
                featureType: 'road',
                elementType: 'geometry',
                stylers: [{
                    color: '#38414e'
                }]
            },
            {
                featureType: 'road',
                elementType: 'geometry.stroke',
                stylers: [{
                    color: '#212a37'
                }]
            },
            {
                featureType: 'road',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#9ca5b3'
                }]
            },
            {
                featureType: 'road.highway',
                elementType: 'geometry',
                stylers: [{
                    color: '#746855'
                }]
            },
            {
                featureType: 'road.highway',
                elementType: 'geometry.stroke',
                stylers: [{
                    color: '#1f2835'
                }]
            },
            {
                featureType: 'road.highway',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#f3d19c'
                }]
            },
            {
                featureType: 'transit',
                elementType: 'geometry',
                stylers: [{
                    color: '#2f3948'
                }]
            },
            {
                featureType: 'transit.station',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#d59563'
                }]
            },
            {
                featureType: 'water',
                elementType: 'geometry',
                stylers: [{
                    color: '#17263c'
                }]
            },
            {
                featureType: 'water',
                elementType: 'labels.text.fill',
                stylers: [{
                    color: '#515c6d'
                }]
            },
            {
                featureType: 'water',
                elementType: 'labels.text.stroke',
                stylers: [{
                    color: '#17263c'
                }]
            }
        ]
    });

    d3.csv("src/data/cleaned/cleanedincidents2020.csv").then(data => {
        let markers = [];
        // let crimeDict = {};
        // let dayCount = 0;
        for (let i = 0; i < data.length; i++) {
            let crime = data[i];
            let date = new Date(crime.dispatch_date_time);
            if (date.getMonth() == 1 && date.getDay() == 1) {
                let marker = crimeMarker(crime.lat, crime.lng);
                markers.push(marker);
                // dayCount++;
                // if (crimeDict.hasOwnProperty(crime.text_general_code)) {
                //     crimeDict[crime.text_general_code]++;
                // } else {
                //     crimeDict[crime.text_general_code] = 1;
                // }

                const {
                    text_general_code: type,
                    dispatch_date_time: time,
                    location_block: location
                } = crime;

                let info = `<b>${location}</b><br>${type}<br> ${time}`;

                marker.addListener("click", () => {
                    infowindow(info).open(map, marker);
                });
            }
        }
        let markerCluster = new MarkerClusterer(map, markers, {
            imagePath: 'https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/m'
        });
        // countCrime(dayCount, crimeDict);
    });
}

const crimeMarker = (lat, lng) => new google.maps.Marker({
    position: new google.maps.LatLng(lat, lng),
    map: map,
    title: "Test point"
});


const infowindow = content => new google.maps.InfoWindow({
    content: content
});

// const countCrime = (count = 0, dict) => {
//     let output = `<h3>Count: ${count}</h3>`;
//     for (let key in dict) {
//         output += `${key}: ${dict[key]}<br>`;
//     }
//     document.getElementById("example").innerHTML = output;
// }