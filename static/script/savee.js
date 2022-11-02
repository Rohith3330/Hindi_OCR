function save(){
    var text = document.getElementById("bar").value;
    text = text.replace(/\n/g, "\r\n"); 
    var blob = new Blob([text], { type: "text/plain"});
    var anchor = document.createElement("a");
    anchor.download = "keyboard_lexilogos.txt";
    anchor.href = window.URL.createObjectURL(blob);
    anchor.target ="_blank";
    anchor.style.display = "none"; // just to be safe
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
 }
 function backspace(){
    var tbInput = document.getElementById("bar");
    tbInput.value = tbInput.value.substr(0, tbInput.value.length - 1);
 }
function clear(){
    document.getElementById("bar").value = "";
}
var doc= new jsPDF();
        $('#cmd').click(function () {
          doc.text($('#content').val(), 10, 10);
          doc.save('Reason.pdf'); 
        });
