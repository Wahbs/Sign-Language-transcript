function changerImage()
        {
             var k=Math.floor(Math.random()*1);
                switch(k)
                    {
                        case 0:document.getElementById('image').src="{% static 'img/Allum.jpg' %}";break
                    }
        }

         function changerImage1()
        {
             var k=Math.floor(Math.random()*2);
                switch(k)
                    {
                        case 1:document.getElementById('image').src="{% static 'img/Eteint.jpg' %}";break
                    }
        }


            const navbar = document.querySelector('.navbar');
            const openNav = document.querySelector('.open-navbar button');
            const closeNav = document.querySelector('.close-navbar');
            const innerUl = document.querySelectorAll('.navbar ul ul');

            function submenuTrigger(e, el){
                e.preventDefault();
                el.classList.toggle('show-submenu');
            }

            innerUl.forEach((el) => {
                if(el.previousElementSibling.nodeName === 'A'){
                    el.previousElementSibling.onclick = (e) => submenuTrigger(e, el);
                }
                el.previousElementSibling.classList.add("submenu-btn");
            });

            openNav.onclick = () => navbar.classList.add('show-navbar');
            closeNav.onclick = () => navbar.classList.remove('show-navbar');
