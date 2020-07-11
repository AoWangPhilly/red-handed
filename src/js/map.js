"use strict";

let map;

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: {
            lat: 39.9509,
            lng: -75.1875
        },
        zoom: 14
    });

    let marker = crimeMarker(39.96233372, -75.16144594);
    google.maps.event.addListener(marker, "click", () => {
        infowindow("<strong>Testing message</strong><br>Yes").open(map, marker);
    });
}

const crimeMarker = (lat, lng) => {
    return new google.maps.Marker({
        position: new google.maps.LatLng(lat, lng),
        map: map,
        title: "Test point"
    });
}

const infowindow = (content) => {
    return new google.maps.InfoWindow({
        content: content
    });
}