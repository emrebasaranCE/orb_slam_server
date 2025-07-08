Server Side

- Flaks ile belirlenen portu dinle. 

Gerekli Fonksiyonlar:

 - Ilk 450 frame"i isleyip ardindan bu degerlerin scale, rotation vb gibi bilgileri kayit eden fonksiyon.
 - Saglik durumu "0" oldlugu durumda gelen frame'leri isleyen ve degerleri kayit eden fonksiyon. - bu fonksiyon bir endpoint ten fotograflarin dosya yolunu bekleyecek.
 - Termal ve normal rgb icin camera parametrelerini dene.



--- 


- baglanti arayuzu kisminda her bir frame in translation verilerini sakladigin bir sistem kur. Ardindan bu verileri 450. frame gelince isle ve orb_slam3 container'i icerisinde scale ve rotation degerlerini bul. Ardindan global bir trajectory bulunmak istendiginde bunlari kolay bir sekilde kullanan endpoint i olustur. Bu endpoint sunlari alacak; 
1 - prev_local_trajectory.
2 - prev_global_trajectory.
3 - cur_local_trajectory.

- takim baglanti arayuzunde herhangi bir frame'i orb_slam3 e gondermek icin slice ve cur_frame file path i alan bir fonksiyon olustur. Bu fonksiyon ardindan gidip baksin bu dosya yoluna ve gerekli slicing islemi kadar frame'i gondersin orb_slam3'e

- kamera parametrelerini degistir.