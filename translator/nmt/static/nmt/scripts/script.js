function show_loader(loaderID){
    var loader = document.getElementById(loaderID);
    loader.style.display = "";
}

function translate_text(){

    var input = document.getElementById('input');
    var output = document.getElementById('output');
    var bleu = document.getElementById('bleu');
    var lang = document.getElementsByClassName('active')[0].title + "_" + document.getElementsByClassName('active')[1].title;

    if (input.value){
        var request = new XMLHttpRequest();
        request.open('GET', 'translate/' + input.value + '/' + lang, false);
        request.send();
    
        var data = JSON.parse(request.responseText);
        output.value = data['trans'];
        if (data['bleu'])
            bleu.innerText = "BLEU SCORE: " + data['bleu'];
        else
            bleu.innerText = "BLEU SCORE"
    }
}

function activateBtn(event){

    if (event.srcElement.parentNode.id == "input-lang"){
        var btns = document.getElementById('input-lang').getElementsByClassName('lang-btn');

        if (event.srcElement.title != 'HR'){
            var out_btns = document.getElementById('output-lang').getElementsByClassName('lang-btn');
            
            for (let index = 0; index < out_btns.length; index++) {
                const element = out_btns[index];
                element.classList.remove('active');
            }
            
            out_btns[0].classList.add('active');
        }
          
    }
    else{
        var btns = document.getElementById('output-lang').getElementsByClassName('lang-btn');
        var input_lang = document.getElementById('input-lang').getElementsByClassName('active');
        
        if (input_lang[0].title != "HR"){
            return;
        }
    }

    for (let index = 0; index < btns.length; index++) {
        const element = btns[index];
        element.classList.remove('active');
    }

    event.srcElement.classList.add('active');
}

