"use strict";

let map;

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: {
            lat: 39.9909,
            lng: -75.1115
        },
        zoom: 11
    });

    d3.csv("src/data/cleanedincidents2020.csv").then(data => {
        let markers = [];
        for (let crime of data) {
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
