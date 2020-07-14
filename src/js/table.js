// https://metadata.phila.gov/#home/datasetdetails/5543868920583086178c4f8e/representationdetails/570e7621c03327dc14f4b68d/
d3.csv("src/data/cleanedincidents2020.csv").then(data => {
    let monthData = [];
    for (const [index, crime] of data.entries()) {
        let date = new Date(crime.dispatch_date_time);
        if (date.getMonth() == 0 && date.getDay() == 3) {
            monthData.push([
                index,
                crime.dc_dist,
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
    $(document).ready(() => {
        $('#datatable').DataTable({
            dom: 'Bfrtip',
            buttons: [
                'copy', 'csv', 'excel', 'pdf', 'print'
            ],
            data: monthData,
            columns: [{
                    title: "#"
                }, {
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
            ],
        });
    });

});