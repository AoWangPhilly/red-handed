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
    let crime = "Aggravated Assault";
    let time = "18:12:00";
    let location = "600 BLOCK WADSWORTH AV";
    let info = `<b>${location}</b><br>${crime}<br> ${time}`;

    marker.addListener("click", () => {
        infowindow(info).open(map, marker);
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