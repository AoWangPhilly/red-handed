"use strict";

var map;

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: {
            lat: 39.9509,
            lng: -75.1575
        },
        zoom: 14
    });
}