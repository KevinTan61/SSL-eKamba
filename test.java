class Animal{
    public void move(){
        System.out.println("动物可以移动");
    }
    public void eat(){
        System.out.println("动物可以吃饭");
    }
}

class Dog extends Animal{
    public void move(int a){
        System.out.println(a);

    }
    public void bark(){
        System.out.println("狗可以吠叫");
    }

}

public class test{
    public static void main(String args[]){
        Animal a = new Animal(); // Animal 对象
        Animal b = new Dog(); // Dog 对象
        Dog d = (Dog) b;
        Dog c = new Dog();
        a.move();// 执行 Animal 类的方法
        b.move();//执行 Dog 类的方法
        b.eat();
        System.out.println("d");
        d.move(1);
        d.move();
        c.move(2);
        System.out.println("c");
        c.move();
       // b.bark();
    }
}