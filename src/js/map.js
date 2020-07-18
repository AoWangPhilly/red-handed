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
        let len = data.length;
        for (let i = 0; i < len; i++) {
            let crime = data[i];
            let date = new Date(crime.dispatch_date_time);
            if (date.getMonth() == 1 && date.getDay() == 1) {
                let marker = crimeMarker(crime.lat, crime.lng);
                markers.push(marker);
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

const getTopFiveCrimes = () => {
    d3.json("src/data/dashboard/typeOfCrimes2019.json").then(crime => {
        let list = "";
        let crimes = Object.keys(crime);
        for (let i = 0; i < 5; i++) {
            list += ` <li class="list-group-item">${i+1}. ${crimes[i]}</li>`
        }
        document.getElementById("topcrime").innerHTML = list;
    });
}

getTopFiveCrimes();