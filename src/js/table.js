d3.csv("src/data/cleanedincidents2020.csv").then(data => {
    let monthData = [];
    for (let crime of data) {
        let date = new Date(crime.dispatch_date_time);
        if (date.getMonth() == 0 && date.getDay() == 3) {
            monthData.push([crime.dc_dist,
                crime.psa,
                crime.dispatch_date_time,
                crime.location_block,
                crime.text_general_code,
                crime.lat,
                crime.lng
            ]);
        } else {
            break;
        }
    }
    console.log(monthData);
    $(document).ready(() => {
        $('#datatable').DataTable({
            data: monthData,
            columns: [{
                    title: "dc_dist"
                },
                {
                    title: "psa"
                },
                {
                    title: "dispatch_date_time"
                },
                {
                    title: "location_block"
                },
                {
                    title: "text_general_code"
                },
                {
                    title: "lat"
                },
                {
                    title: "lng"
                }
            ]
        });
    });

});