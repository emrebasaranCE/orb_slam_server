Server Side

- Flaks ile belirlenen portu dinle. 

Gerekli Fonksiyonlar:

 - Ilk 450 frame"i isleyip ardindan bu degerlerin scale, rotation vb gibi bilgileri kayit eden fonksiyon.
 - Saglik durumu "0" oldlugu durumda gelen frame'leri isleyen ve degerleri kayit eden fonksiyon. - bu fonksiyon bir endpoint ten fotograflarin dosya yolunu bekleyecek.
 - Termal ve normal rgb icin camera parametrelerini dene.



--- 

local trajectory islemini, globale cevirmem lazim. Bunu da islenicek frameden once olan frame'i bir kez orb_slam den gecirerek local degerini bulup, buradan da bulmak istedigim frame'in  trajectory degerini bulabiliriz.